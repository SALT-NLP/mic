# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
)


class CustomDataset(Dataset):
    def __init__(
        self, data, maxlen, with_labels=True, bert_model="albert-base-v2", lbl="label"
    ):

        self.data = data  # pandas dataframe
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen
        self.with_labels = with_labels
        self.lbl = lbl

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, "sentence1"])
        sent2 = str(self.data.loc[index, "sentence2"])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(
            sent1,
            sent2,
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate to max_length
            max_length=self.maxlen,
            return_tensors="pt",
        )  # Return torch.Tensor objects

        token_ids = encoded_pair["input_ids"].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair["attention_mask"].squeeze(
            0
        )  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair["token_type_ids"].squeeze(
            0
        )  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, self.lbl]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


class SentencePairClassifier(nn.Module):
    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
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
        elif bert_model in {
            "bert-base-uncased",
            "roberta-base-uncased",
        }:  # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        """
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        """

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output = self.bert_layer(
            input_ids, attn_masks, token_type_ids
        )

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits


def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for _, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = (
                seq.to(device),
                attn_masks.to(device),
                token_type_ids.to(device),
                labels.to(device),
            )
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count


def train_bert(args, criterion, train_loader, val_loader, save_model=True):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch

    epochs = args.epochs
    iters_to_accumulate = args.iters_to_accumulate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SentencePairClassifier(args.bert_model, freeze_bert=args.freeze_bert)

    if (torch.cuda.device_count() > 1) and (args.gpu == -1):  # if multiple GPUs
        print("Using", torch.cuda.device_count(), "GPUs.")
        net = nn.DataParallel(net)

    net.to(device)

    opti = AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    num_warmup_steps = 0  # The number of steps for the warmup phase.
    t_total = (
        len(train_loader) // args.iters_to_accumulate
    ) * args.epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(
            tqdm(train_loader)
        ):

            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, labels = (
                seq.to(device),
                attn_masks.to(device),
                token_type_ids.to(device),
                labels.to(device),
            )

            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits.squeeze(-1), labels.float())
                loss = (
                    loss / iters_to_accumulate
                )  # Normalize the loss because it is averaged

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
                print(
                    "Iteration {}/{} of epoch {} complete. Loss : {} ".format(
                        it + 1, nb_iterations, ep + 1, running_loss / print_every
                    )
                )

                running_loss = 0.0

        if not args.test:
            out_prefix = os.path.join(
                args.output,
                "%s-%s-%s-%s-%s"
                % (
                    args.bert_model,
                    args.label,
                    args.batchsize,
                    ep + 1,
                    int(args.lr * 1e5),
                ),
            )
            predict_and_save(
                args,
                net,
                device,
                val_loader,
                with_labels=True,
                out_prefix=out_prefix,
                save_model=False,
            )

        val_loss = evaluate_loss(
            net, device, criterion, val_loader
        )  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep + 1, val_loss))

        if val_loss < best_loss:
            print(
                "Best validation loss improved from {} to {}".format(
                    best_loss, val_loss
                )
            )
            print()
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    out_prefix = os.path.join(
        args.output,
        "%s-%s-%s-%s-%s"
        % (
            args.bert_model,
            args.label,
            args.batchsize,
            args.epochs,
            int(args.lr * 1e5),
        ),
    )
    predict_and_save(
        args,
        net,
        device,
        val_loader,
        with_labels=True,
        out_prefix=out_prefix,
        save_model=save_model,
    )

    del loss
    torch.cuda.empty_cache()
    return out_prefix + ".pt"


def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()


def test_prediction(
    net, device, dataloader, with_labels=True, result_file="results/output.txt"
):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    with open(result_file, "w") as w:
        probs_all = []

        with torch.no_grad():
            if with_labels:
                for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                    seq, attn_masks, token_type_ids = (
                        seq.to(device),
                        attn_masks.to(device),
                        token_type_ids.to(device),
                    )
                    logits = net(seq, attn_masks, token_type_ids)
                    probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                    probs_all += probs.tolist()
            else:
                for seq, attn_masks, token_type_ids in tqdm(dataloader):
                    seq, attn_masks, token_type_ids = (
                        seq.to(device),
                        attn_masks.to(device),
                        token_type_ids.to(device),
                    )
                    logits = net(seq, attn_masks, token_type_ids)
                    probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                    probs_all += probs.tolist()

        w.writelines(str(prob) + "\n" for prob in probs_all)

def predict_and_save(
    args, model, device, dataloader, with_labels, out_prefix, save_model=True
):
    if save_model:
        torch.save(model.state_dict(), out_prefix + ".pt")
        print("The model has been saved in {}".format(out_prefix + ".pt"))

    path_to_output_file = (
        out_prefix + "_output.txt"
    )  # path to the file with prediction probabilities

    print("Predicting on test data...")
    test_prediction(
        net=model,
        device=device,
        dataloader=val_loader,
        with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
        result_file=path_to_output_file,
    )
    print()
    print("Predictions are available in : {}".format(path_to_output_file))

    y_test = df_val[args.label]  # true labels

    probs_test = pd.read_csv(path_to_output_file, header=None)[
        0
    ]  # prediction probabilities
    threshold = 0.5  # you can adjust this threshold for your own dataset
    y_pred = (probs_test >= threshold).astype(
        "uint8"
    )  # predicted labels using the above fixed threshold

    results = {
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
        "label": args.label,
        "max_len": args.maxlen,
        "batchsize": args.batchsize,
        "epochs": args.epochs,
        "iters_to_accumulate": args.iters_to_accumulate,
        "lr": args.lr,
        "epsilon": args.epsilon,
        "seed": args.seed,
    }
    for label in sorted(set(y_test)):
        results["precision_%s" % label] = precision_score(
            y_test, y_pred, labels=[label], average="macro"
        )
        results["recall_%s" % label] = recall_score(
            y_test, y_pred, labels=[label], average="macro"
        )
        results["f1_%s" % label] = f1_score(
            y_test, y_pred, labels=[label], average="macro"
        )

    print(results)

    with open(out_prefix + ".json", "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        type=str,
        default="label",
        help="the y label for the dataframe column to predict",
    )
    parser.add_argument("--gpu", type=int, default=0, choices=list(range(8)))
    parser.add_argument(
        "--input",
        type=str,
        default="splits",
        help="path to the directory of input train/dev/test splits",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="path to directory for outputting results",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=512,
        help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded',
    )
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--save_model_every",
        type=int,
        default=0,
        help="save the model checkpoint every N epochs",
    )
    parser.add_argument(
        "--iters_to_accumulate",
        type=int,
        default=2,
        help='the gradient accumulation adds gradients over an effective batch of size : args.batch_size * iters_to_accumulate. If set to "1", you get the usual batch size',
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--bert_model",
        type=str,
        default="albert-base-v2",
        choices=[
            "albert-base-v2",
            "albert-large-v2",
            "albert-xlarge-v2",
            "albert-xxlarge-v2",
            "bert-base-uncased",
        ],
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        help="if True, freeze the encoder weights and only update the classification layer weights",
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--loss_weight",
        type=float,
        default=0,
        help="positive class weights for the loss",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed for replicability"
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    df_train = pd.read_csv(os.path.join(args.input, "train.csv"))
    df_val = pd.read_csv(os.path.join(args.input, "dev.csv"))
    df_test = pd.read_csv(os.path.join(args.input, "test.csv"))

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
    train_set = CustomDataset(df_train, args.maxlen, args.bert_model, lbl=args.label)
    print("Reading validation data...")
    val_set = CustomDataset(df_val, args.maxlen, args.bert_model, lbl=args.label)
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batchsize, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=args.batchsize, num_workers=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = df_train[args.label]
    weights = [
        (len(y_train) - sum(y_train)) / sum(y_train)
    ]  # len(y_train) / (len(set(y_train)) * np.bincount(y_train))
    weights = torch.FloatTensor(weights).cuda(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    if args.loss_weight:
        criterion = nn.BCEWithLogitsLoss(
            torch.FloatTensor([args.loss_weight]).cuda(device)
        )

    path_to_model = train_bert(
        args, criterion, train_loader, val_loader, save_model=args.test
    )  # only save if testing
