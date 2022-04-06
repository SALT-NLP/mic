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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score, mean_squared_error
from ast import literal_eval
from glob import glob

import random
import pickle as pkl
from collections import Counter
import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, accuracy_score, mean_squared_error
from ast import literal_eval
from glob import glob
from scipy.stats import pearsonr

import random
import pickle as pkl
from collections import Counter

def condense_label_classes(y, labels, threshold=0.5):
    """
    Takes an m x n matrix @y and a list of label classes @labels of length n
    and "condenses" @y, taxing argmax for any labels that belong to the same class. 
    For example, with @labels = [0, 0, 0, 1, 1] and @y = [[0.2, 0.5, 0.8, 0.2, 0.6], ...]
    the function would return [[2, 1], ...] because relative index 2 is argmax for label class 0 
    and relative index 1 is argmax for label class 1
    
    For label classes with only one label, we treat this as the binary case: convert to int
    by a comparison with a @threshold
    """
    
    y_transformed = []
    for label in sorted(set(labels)):
        y_l = y[:, (np.array(labels)==label)]

        if y_l.shape[-1]>1:
            y_l = np.argmax(y_l, axis=1)
        else:
            y_l = (y_l>threshold).astype(int).flatten()

        y_transformed.append(y_l)
    
    return np.stack(y_transformed).T

def build_results_dict_regression(args, y_pred_continuous, y_test):
    y_pred_continuous = y_pred_continuous.T[0]
    y_pred = np.rint(y_pred_continuous).astype(int)
    r, p = pearsonr(y_test, y_pred_continuous)
    results = {
        'label': args.label,
        'MSE': mean_squared_error(y_test, y_pred_continuous),
        'pearson_r': r,
        'pearson_p_value': p,
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
    }

    for label in set(y_test):
        results[f"precision_{label}"] = precision_score(y_test, y_pred, labels=[label], average='macro')
        results[f"recall_{label}"] = recall_score(y_test, y_pred, labels=[label], average='macro')
        results[f"f1_{label}"] = f1_score(y_test, y_pred, labels=[label], average='macro')
        
        results[f"accuracy_{label}"] = accuracy_score(y_test[y_test==label], y_pred[y_test==label])

    return results

def build_results_dict_multilabel(args, y_pred, y_test):
    results = {
        'label': args.label
    }
    precisions = []
    recalls = []
    f1s = []
    for class_idx in range(y_test.shape[1]):
        P = precision_score(y_test[:, class_idx], y_pred[:, class_idx], average='macro')
        R = recall_score(y_test[:, class_idx], y_pred[:, class_idx], average='macro')
        F = f1_score(y_test[:, class_idx], y_pred[:, class_idx], average='macro')
        results[f"precision_{class_idx}_macro"] = P
        results[f"recall_{class_idx}_macro"] = R
        results[f"f1_{class_idx}_macro"] = F
        
        precisions.append(P)
        recalls.append(R)
        f1s.append(F)
        
        results[f"accuracy_{class_idx}"] = accuracy_score(y_test[:, class_idx], y_pred[:, class_idx])
        for label in set(y_test[:, class_idx]):
            results[f"precision_{class_idx}_{label}"] = precision_score(y_test[:, class_idx], y_pred[:, class_idx], labels=[label], average='macro')
            results[f"recall_{class_idx}_{label}"] = recall_score(y_test[:, class_idx], y_pred[:, class_idx], labels=[label], average='macro')
            results[f"f1_{class_idx}_{label}"] = f1_score(y_test[:, class_idx], y_pred[:, class_idx], labels=[label], average='macro')   
            
        
    results[f"precision_macro"] = np.mean(np.array(precisions))
    results[f"recall_macro"] = np.mean(np.array(recalls))
    results[f"f1_macro"] = np.mean(np.array(f1s))
    return results

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2', sent1='sentence1', sent2='sentence2', lbl='label', eval_list_literal=False):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 
        self.lbl = lbl
        self.sent1 = sent1
        self.sent2 = sent2
        
        if eval_list_literal:
            lst = []
            for x in self.data[self.lbl].values:
                try: 
                    lst.append(literal_eval(x))
                except:
                    print(x)

            self.data[self.lbl] = lst

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        encoded = {}
        
        if self.sent2:
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
        attn_masks = encoded_pair['attention_mask'].squeeze(0) # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0) # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels       
            label = torch.tensor(self.data.loc[index, self.lbl], dtype=torch.float)
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
        elif bert_model in {"bert-base-uncased", "roberta-base"}: # 110M parameters
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
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count


def build_backtr(fn, target):
    backtr = pkl.load(open(fn, 'rb'))
    return {backtr[key][target]: backtr[key] for key in backtr}

def build_pool(df, target_col, backtr):
    pool = []
    for _, row in df.iterrows():
        target = row[target_col] 
        
        for translation in backtr[target]['translations']:
            row_ = row.copy()
            row_[target_col] = translation
            pool.append(row_)
    return pool

def balance(df, balance_col, target, backtr_fn, random_state=0):
    random.seed(random_state)
    augmentation = []
    backtr = build_backtr(backtr_fn, target)
    c = Counter(df[balance_col].values)
    largest_class, largest_class_size = c.most_common()[0]
    for label in c.keys():
        if label != largest_class:
            pool = build_pool(df[df[balance_col]==label], target, backtr)
            augmentation.extend(random.choices(pool, k=largest_class_size-c[label]))
    return pd.concat([pd.DataFrame(augmentation), df]).sample(frac=1, random_state=random_state)

def test_prediction(net, device, dataloader, with_labels=True, use_sigmoid=True):
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
                    torch.sigmoid(logits).detach().cpu().numpy() if use_sigmoid else logits.detach().cpu().numpy()
                )
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs_all.append(
                    torch.sigmoid(logits).detach().cpu().numpy() if use_sigmoid else logits.detach().cpu().numpy()
                )

    return np.concatenate(probs_all, axis=0)

def predict_and_save(args, model, device, df_test, dataloader, with_labels, out_prefix, save_model=True, multilabel=False, is_regression=True): 
    if save_model:
        torch.save(model.state_dict(), out_prefix+".pt")
        #model.save_pretrained(out_prefix)
        print("The model has been saved in {}".format(out_prefix+".pt"))
    
    path_to_output_file = out_prefix+"_output.txt" # path to the file with prediction probabilities

    print("Predicting on test data...")
    y_pred = test_prediction(net=model, device=device, dataloader=dataloader, with_labels=True, use_sigmoid=(not is_regression))
    
    results = {}
    if is_regression:
        y_test = np.array(df_test[args.label].values)  # true labels
        results = build_results_dict_regression(args, y_pred, y_test)
    else:
        np.save(out_prefix+'_raw_probas.npy', np.array(y_pred))
        threshold = 0.5   # adjust this threshold?
        y_pred = (y_pred>=threshold).astype('uint8')

        y_test = np.array(df_test[args.label].values.tolist())  # true labels

        if len(args.label_classes):
            y_pred = condense_label_classes(y_pred, args.label_classes, threshold=threshold)
            y_test = condense_label_classes(y_test, args.label_classes, threshold=threshold)
        results = build_results_dict_multilabel(args, y_pred, y_test)

    print(results)
    
    with open( out_prefix+'.json', 'w') as outfile:
        json.dump(results, outfile)
        
    np.save(out_prefix+'_predictions.npy', np.array(y_pred))
    torch.save(args, out_prefix + "_training_args.bin")

def train_bert(net, args, criterion, df_train, df_val, train_loader, val_loader, save_model=True, is_regression=True):

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
                loss = criterion(logits.squeeze(-1), labels.float())
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
            predict_and_save(args, net, device, df_val, val_loader, with_labels=True, out_prefix=out_prefix, save_model=False, is_regression=is_regression)

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
    predict_and_save(args, net, device, df_val, val_loader, with_labels=True, out_prefix=out_prefix, save_model=save_model, is_regression=is_regression)

    del loss
    torch.cuda.empty_cache()
    return out_prefix+".pt"