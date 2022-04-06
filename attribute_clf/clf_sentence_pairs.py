import torch
import torch.nn as nn
import os, json
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2', sent1='sentence1', sent2='sentence2', lbl='label'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 
        self.lbl = lbl
        self.sent1 = sent1
        self.sent2 = sent2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        encoded = {}
        
        if args.sent2:
            sent1 = str(self.data.loc[index, self.sent1])
            sent2 = str(self.data.loc[index, self.sent2])

            # Tokenize the pair of sentences to get token ids, attention masks and token type ids
            encoded_pair = self.tokenizer(sent1, sent2, 
                                          padding='max_length',  # Pad to max_length
                                          truncation=True,  # Truncate to max_length
                                          max_length=self.maxlen,  
                                          return_tensors='pt')  # Return torch.Tensor objects
        else:
            sent1 = str(self.data.loc[index, self.sent1])
            encoded_pair = self.tokenizer.encode_plus(sent1, 
                                            add_special_tokens=True, 
                                            max_length=self.maxlen,
                                            truncation=True, # Truncate to max_length
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt' # Return torch.Tensor objects
                                           )
            


        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, self.lbl].astype(int)
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids
        
class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False, labels_count=1):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            hidden_size = 4096
        elif bert_model in {"bert-base-uncased", 'roberta-base'}: # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, labels_count)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits
    
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits, labels).item()
            count += 1

    return mean_loss / count

def train_bert(net, args, criterion, train_loader, val_loader, save_model=True):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []
    lr = args.lr
    epochs = args.epochs
    iters_to_accumulate = args.iters_to_accumulate
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (torch.cuda.device_count() > 1) and (args.gpu==-1):  # if multiple GPUs
        print("Using", torch.cuda.device_count(), "GPUs.")
        net = nn.DataParallel(net)

    net.to(device)
    
    opti = AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = args.epochs * len(train_loader)  # The total number of training steps
    t_total = (len(train_loader) // args.iters_to_accumulate) * args.epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits, labels)
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()

            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0

        if not args.test:
            out_prefix = os.path.join(args.output, "%s-%s-%s-%s-%s" % (args.bert_model, args.label, args.batchsize, ep+1, int(args.lr*1e5)))
            predict_and_save(args, net, device, val_loader, with_labels=True, out_prefix=out_prefix, save_model=False)

        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    out_prefix = os.path.join(args.output, "%s-%s-%s-%s-%s" % (args.bert_model, args.label, args.batchsize, args.epochs, int(args.lr*1e5)))
    predict_and_save(args, net, device, val_loader, with_labels=True, out_prefix=out_prefix, save_model=save_model)

    del loss
    torch.cuda.empty_cache()
    return out_prefix+".pt"

def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and return the results
    """
    net.eval()
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs_all.append(
                    logits.detach().cpu().numpy()
                )
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs_all.append(
                    logits.detach().cpu().numpy()
                )

    return np.concatenate(probs_all, axis=0)
    
def predict_and_save(args, model, device, dataloader, with_labels, out_prefix, save_model=True, multilabel=False): 
    if save_model:
        torch.save(model.state_dict(), out_prefix+".pt")
        print("The model has been saved in {}".format(out_prefix+".pt"))

    print("Predicting on test data...")
    probs = test_prediction(net=model, device=device, dataloader=val_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                    result_file=path_to_output_file)
    y_pred = np.argmax(probs, axis=1).flatten()
    if multilabel:
        threshold = 0.5   # adjust this threshold?
        y_pred = (probs>=threshold).astype('uint8')
        
    np.save(out_prefix+'_raw_probas.npy', probs)

    y_test = df_val[args.label]  # true labels
    
    results = {
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'accuracy': accuracy_score(y_test, y_pred),
        'label': args.label,
        'max_len': args.maxlen,
        'batchsize': args.batchsize,
        'epochs': args.epochs,
        'iters_to_accumulate': args.iters_to_accumulate,
        'lr': args.lr,
        'epsilon': args.epsilon,
        'seed': args.seed
    }
    for label in sorted(set(y_test)):
        results['precision_%s' % label] = precision_score(y_test, y_pred, labels=[label], average='macro')
        results['recall_%s' % label] = recall_score(y_test, y_pred, labels=[label], average='macro')
        results['f1_%s' % label] = f1_score(y_test, y_pred, labels=[label], average='macro')

    print(results)
    
    with open( out_prefix+'.json', 'w') as outfile:
        json.dump(results, outfile)
        
    np.save(out_prefix+'_predictions.npy', y_pred)
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='label', help='the y label for the dataframe column to predict')
    parser.add_argument('--sent1', type=str, help='the label for the dataframe column to set as sentence 1')
    parser.add_argument('--sent2', type=str, help='the label for the dataframe column to set as sentence 2')
    parser.add_argument('--gpu', type=int, default=0, choices=list(range(8)))
    parser.add_argument('--input', type=str, default='splits', help='path to input file')
    parser.add_argument('--output', type=str, default = 'results', help='path to directory for outputting results')
    parser.add_argument('--maxlen', type=int, default=512, help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--save_model_every', type=int, default=0, help='save the model checkpoint every N epochs')
    parser.add_argument('--iters_to_accumulate', type=int, default=2, help='the gradient accumulation adds gradients over an effective batch of size : args.batch_size * iters_to_accumulate. If set to "1", you get the usual batch size')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--bert_model', type=str, default='albert-base-v2', choices=['albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', 'roberta-base'])
    parser.add_argument('--freeze_bert', action='store_true', help="if True, freeze the encoder weights and only update the classification layer weights")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--loss_weight', type=float, default=0, help='positive class weights for the loss')
    parser.add_argument('--seed', type=int, default=1, help='random seed for replicability')
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    df = pd.read_csv(args.input)
    df_train = df[df['split']=='train'].copy().reset_index()
    df_val = df[df['split']=='dev'].copy().reset_index()
    df_test = df[df['split']=='test'].copy().reset_index()
    del df
    
    if args.test:
        df_train = pd.concat([df_train, df_val]).reset_index()
        df_val = df_test.copy()
        
    print(df_train.head())

    #  Set all seeds to make reproducible results
    set_seed(args.seed)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Creating instances of training and validation set
    print("Reading training data...")
    train_set = CustomDataset(df_train, args.maxlen, args.bert_model, sent1=args.sent1, sent2=args.sent2, lbl=args.label)
    print("Reading validation data...")
    val_set = CustomDataset(df_val, args.maxlen, args.bert_model, sent1=args.sent1, sent2=args.sent2, lbl=args.label)
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batchsize, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=args.batchsize, num_workers=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = df_train[args.label]
    weights = len(y_train) / (len(set(y_train)) * np.bincount(y_train))    
    #weights = np.tile(weights,args.batchsize).reshape(args.batchsize,-1).T
    weights = torch.FloatTensor(weights).cuda(device).half()
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = SentencePairClassifier(args.bert_model, freeze_bert=args.freeze_bert, labels_count=len(set(y_train)))
    path_to_model = train_bert(model, args, criterion, train_loader, val_loader, save_model=args.test) # only save if testing
    