# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        maxlen,
        with_labels=True,
        bert_model="albert-base-v2",
        sentence1="sentence1",
        sentence2="sentence2",
    ):

        self.data = data  # pandas dataframe
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, args.sent1])  #'questions'
        sent2 = str(self.data.loc[index, args.sent2])  #'blenderbot_A0'

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
            label = self.data.loc[index, "label"]
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
        elif bert_model == "bert-base-uncased":  # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, choices=list(range(8)))
    parser.add_argument("--input", type=str, default="input.csv", help="input file")
    parser.add_argument("--output", type=str, default="output.csv", help="output file")
    parser.add_argument(
        "--sent1",
        type=str,
        default="questions",
        help="the name of the column for sentence 1",
    )
    parser.add_argument(
        "--sent2",
        type=str,
        default="blenderbot_A0",
        help="the name of the column for sentence 2",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=512,
        help='maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded',
    )
    parser.add_argument("--batchsize", type=int, default=16)
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
        "--path_to_model", type=str, help="path to the trained bert model"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed for replicability"
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.input)

    set_seed(args.seed)

    print("Reading test data...")
    test_set = CustomDataset(
        df, args.maxlen, bert_model=args.bert_model, with_labels=False
    )
    test_loader = DataLoader(test_set, batch_size=args.batchsize, num_workers=5)

    model = SentencePairClassifier(args.bert_model)
    if (torch.cuda.device_count() > 1) and (args.gpu == -1):  # if multiple GPUs
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)

    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(args.path_to_model))
    model.to(device)

    print("Predicting on test data...")
    test_prediction(
        net=model,
        device=device,
        dataloader=test_loader,
        with_labels=False,
        result_file=args.output,
    )
    print()
    print("Predictions are available in : {}".format(args.output))
