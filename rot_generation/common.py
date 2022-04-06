import pandas as pd
from transformers import Trainer
import argparse
from datasets import load_dataset, load_metric
import pandas as pd
from collections import OrderedDict, Counter
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator, Union
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, Trainer
import nltk, os, csv, json, random
import numpy as np
import torch
from tqdm import tqdm
import json


ATTR_MAPPINGS = {
    'AGREE_TO_STR': OrderedDict([(1, "nobody"), (2, "rare"), (3, "controversial"), (4, "most"), (5, "all")]),
    'VIOLATION_SEVERITY_TO_STR': OrderedDict([(1, "fine"), (2, "unwise"), (3, "bad"), (4, "horrible"), (5, "worst")]),
    'ALIGNMENT_TO_STR': OrderedDict([(0, "disagrees"), (1, "neutral"), (2, "agrees")])
}

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_all_attributes(df):    
    attributes = set()
    for _, row in df.iterrows():
        if 'moral' in row:
            check = 'moral'
        elif 'rot-moral-foundations' in row:
            check = 'rot-moral-foundations'
        else:
            continue
        if pd.notna(row[check]) and not len(row[check])==0:
            for category in sorted(row[check].split("|")):
                attributes.add(f"<{category}>")
                    
    for mapping_key in ATTR_MAPPINGS:
        for key in ATTR_MAPPINGS[mapping_key]:
            attributes.add(f"<{ATTR_MAPPINGS[mapping_key][key]}>")
    return list(attributes)

def init_attribute_embeddings(model, tokenizer, special_tokens):
    """
    Initialize each attribute embedding (e.g. <very-bad>) with the bag of words of its words (vec(very) + vec(bad))
    """
    embeddings = model.get_input_embeddings()
    unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    for word in special_tokens:
        index = tokenizer.convert_tokens_to_ids(word)

        if word.startswith("<"):
            other_words = word[1:-1].replace("-", " ").split()
            other_indices = tokenizer.convert_tokens_to_ids(other_words)
            other_indices = [i for i in other_indices if i != unk_token_id]
            if len(other_indices) == 0:
                continue
            elif len(other_indices) == 1:
                vec = embeddings.weight.data[other_indices[0], :]
            else:
                vec = torch.sum(
                    torch.stack([embeddings.weight.data[i, :] for i in other_indices])
                )

            embeddings.weight.data[index, :] = vec

    model.set_input_embeddings(embeddings)

def get_attr(row: pd.Series, col: str) -> List[str]:
    res: List[str] = []

    if col == "moral":
        if pd.notna(row["moral"]) and not len(row["moral"])==0:
            # Multi-label
            for category in sorted(row["moral"].split("|")):
                res.append(f"<{category}>")
    elif col == "rot-moral-foundations":
        if pd.notna(row["rot-moral-foundations"]) and not len(row["rot-moral-foundations"])==0:
            for category in sorted(row["rot-moral-foundations"].split("|")):
                res.append(f"<{category}>")
    elif col == "rot-agree":
        if pd.notna(row["rot-agree"]):
            res.append(f"<{ATTR_MAPPINGS['AGREE_TO_STR'][row['rot-agree']]}>")
    elif col == "violation-severity":
        if pd.notna(row["violation-severity"]):
            res.append(f"<{ATTR_MAPPINGS['VIOLATION_SEVERITY_TO_STR'][row['violation-severity']]}>")
    elif col == "A_agrees":
        if pd.notna(row['A_agrees']):
            res.append(f"<{ATTR_MAPPINGS['ALIGNMENT_TO_STR'][row['A_agrees']]}>")
    else:
        raise ValueError(f"Unknown attribute: '{col}'")

    return res

def get_rot_attributes(row: pd.Series) -> List[str]:
    """
    Gets a row from the rot-details tsv file and returns a list of string rot-related attributes
    :param row: dataframe row
    :return: a list of string rot-related attributes
    """
    return (
        get_attr(row, "A_agrees")
        + get_attr(row, "rot-agree")
        + get_attr(row, "violation-severity")
        + get_attr(row, "moral")
    )

def build(row, format_string):
    formats = format_string.split("~")
    outputs = []
    for form in formats:
        
        output = []
        for element in form.split():
            if '[' in element:
                output.append(element)
            elif '<' in element:
                output.append(' '.join(get_attr(row, element.replace('<', '').replace('>', ''))))
            else:
                output.append(row[element].strip())
        outputs.append(" ".join(output))
    outputs[-1] += " <eos>"
    return outputs

def tokenize(string, tokenizer, eos_id=None):
    return tokenizer(string)
#     if not eos_id:
#         eos_id = tokenizer.eos_token_id
    
#     t = tokenizer(string)
#     input_ids = t['input_ids']
#     if type(input_ids[0])==list:
#         input_ids = [ np.array(row)[np.array(row)!=eos_id] for row in input_ids]
#     else:
#         input_ids = np.array(input_ids)
#         input_ids = input_ids[input_ids!=eos_id]
#     t['input_ids'] = input_ids
#     return t

def preprocess(examples, tokenizer, format_string):
    source_target = [build(row, format_string)
                  for _, row in pd.DataFrame(examples).iterrows()]
    source = [tup[0] for tup in source_target]
    target = [tup[1] if len(tup)>1 else "" for tup in source_target]
    
    model_inputs = tokenize(source, tokenizer) #tokenizer(source)

    with tokenizer.as_target_tokenizer():
        labels = tokenize(target, tokenizer) #tokenizer(target)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def decode(args, df, model, tokenizer, skip_special_tokens=True, remove_history=False):
    model = model
    is_greedy = (args.top_p == 0) and (args.top_k == 0) and (args.beams == 0)
    
    eos_token_id = tokenizer.encode("<eos>", add_special_tokens=False)[0]
    generations = []
    
    for _, row in df.iterrows():
        input_ids = torch.tensor([row['input_ids']], device='cuda')
        
        out = model.generate(
                input_ids,
                do_sample=args.beams == 0,
                max_length=args.maxlen,
                temperature=args.temperature,
                top_p=args.top_p if args.top_p > 0 else None,
                top_k=args.top_k if args.top_k > 0 else None,
                num_beams=args.beams if args.beams > 0 else None,
                early_stopping=True,
                pad_token_id=50256,
                no_repeat_ngram_size=3,
                eos_token_id=eos_token_id
            )
        if remove_history:
            generations.append(tokenizer.decode(out[:, input_ids.shape[-1]:][0], skip_special_tokens=skip_special_tokens))
        
        else:
            generations.append(tokenizer.decode(out[0],
                                            skip_special_tokens=skip_special_tokens
                                           ))
    return generations