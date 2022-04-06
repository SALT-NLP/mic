# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import os
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    pipeline,
)


def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def modify_prompt(prompt, person_A, person_B):

    new_prompt = f"""
The following is a conversation between {person_A} and {person_B}.

{person_A}: {prompt}

{person_B}:"""

    return new_prompt


def extract_main_response(response, stop_person):
    return (
        response[0]["generated_text"].split("\n")[0].strip()
    )  # .split('%s:' % stop_person)[0]


def run_gpt_neo(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print("loading generator...")
    generator = pipeline(
        "text-generation", model="EleutherAI/gpt-neo-2.7B", device=args.gpu
    )

    for k, input in enumerate(
        list(glob( os.path.join(args.indir, '*.csv') ))[
            args.start_idx : args.end_idx
        ]
    ):

        print("starting batch %s / %s" % (k, (args.end_idx - args.start_idx)))
        output = os.path.join(args.outdir, os.path.basename(input))

        if os.path.exists(output):
            print("already saved", output)
            continue

        df = pd.read_csv(input)

        answers = []
        names_1 = []
        names_2 = []
        for initial_question in tqdm(df.questions.values):
            answer = ""

            person_A = random.choice(sorted(args.baby_names))
            person_B = random.choice(sorted(args.baby_names.difference(set(person_A))))

            new_prompt = modify_prompt(initial_question, person_A, person_B)
            out = generator(
                new_prompt, min_length=100, max_length=200, return_full_text=False
            )
            answer = extract_main_response(out, person_B)
            if args.verbose:
                print(
                    "Q: ",
                    initial_question,
                    "\nA: ",
                    answer,
                    "\nperson_A: ",
                    person_A,
                    "\nperson_B: ",
                    person_B,
                )
            answers.append(answer)
            names_1.append(person_A)
            names_2.append(person_B)

        df["gpt-neo_A0"] = answers
        df["person_A"] = names_1
        df["person_B"] = names_2
        df.to_csv(output, index=False)


def run_blenderbot(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu

    mname = "facebook/blenderbot-3B"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)

    model.to(device)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    for k, input in enumerate(
        list(glob( os.path.join(args.indir, '*.csv') ))[
            args.start_idx : args.end_idx
        ]
    ):

        print("starting batch %s / %s" % (k, (args.end_idx - args.start_idx)))
        output = os.path.join(args.outdir, os.path.basename(input))

        if os.path.exists(output):
            print("already saved", output)
            continue

        df = pd.read_csv(input)

        answers = []
        if args.batch:
            inputs = tokenizer(
                list(df.questions.values), return_tensors="pt", padding=True
            ).to(device)
            reply_ids = model.generate(**inputs)
            answers = tokenizer.batch_decode(reply_ids)
        else:
            for initial_question in tqdm(df.questions.values):
                answer = ""
                try:
                    inputs = tokenizer([initial_question], return_tensors="pt").to(
                        device
                    )
                    reply_ids = model.generate(**inputs)
                    answer = (
                        tokenizer.batch_decode(reply_ids)[0]
                        .replace("<s> ", "")
                        .replace(" </s>", "")
                        .replace("<s>", "")
                        .replace("</s>", "")
                    )
                except Exception:
                    pass
                if args.verbose:
                    print("Q: ", initial_question, "\nA: ", answer)
                answers.append(answer)

        df["blenderbot_A0"] = answers
        df.to_csv(output, index=False)


def run_dialoGPT(args):
    for input in reversed(list(glob( os.path.join(args.indir, '*.csv') ))):

        output = os.path.join(args.outdir, os.path.basename(input))

        df = pd.read_csv(input)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

        follow_up_questions = ["Why"]

        answers = {i: [] for i in range(len(follow_up_questions) + 1)}

        chat_history_ids = []
        for initial_question in tqdm(df.questions.values):

            for i, q in enumerate([initial_question] + follow_up_questions):
                input_ids = tokenizer.encode(
                    q + tokenizer.eos_token, return_tensors="pt"
                )

                concat_input_ids = (
                    torch.cat([chat_history_ids, input_ids], dim=-1)
                    if i > 0
                    else input_ids
                )

                chat_history_ids = model.generate(
                    concat_input_ids,
                    max_length=1000,
                    pad_token_id=tokenizer.eos_token_id,
                )

                answer = tokenizer.decode(
                    chat_history_ids[:, concat_input_ids.shape[-1] :][0],
                    skip_special_tokens=True,
                )
                answers[i].append(answer)

        for i in range(len(answers)):
            df["dialoGPT_A%s" % i] = answers[i]

        df.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, choices=["blenderbot", "gpt-neo", "dialoGPT"]
    )
    parser.add_argument("--seed", type=int, default=38)

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=int(1e10))
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--baby_names",
        type=set,
        default=set(
            pd.read_csv("resources/boy_names_2018.csv")[
                "Name"
            ]
        ).union(
            set(
                pd.read_csv("resources/girl_names_2018.csv")[
                    "Name"
                ]
            )
        ),
    )
    parser.add_argument("--indir", type=str, default='data/raw/Reddit-QA-Corpus/filtered/')
    parser.add_argument("--outdir", type=str, default="data/prepared/blenderbot_generations/")

    args = parser.parse_args()

    set_seed(args.seed)

    run_model = run_dialoGPT
    if args.model_name == "blenderbot":
        run_model = run_blenderbot
    elif args.model_name == "gpt-neo":
        run_model = run_gpt_neo

    run_model(args)
