The Moral Integrity Corpus (MIC)
=================================

This repository contains source code for the **The Moral Integrity Corpus: A Benchmark for Ethical Dialogue Systems** by [Caleb Ziems](https://calebziems.com/), Jane A. Yu, [Yi-Chia Wang](https://scholar.google.com/citations?user=9gMgFPQAAAAJ&hl=en), [Alon Y. Halevy](https://scholar.google.com/citations?user=F_MI0pcAAAAJ&hl=en), [Diyi Yang](https://www.cc.gatech.edu/~dyang888/)

[[Paper]](https://faculty.cc.gatech.edu/~dyang888/docs/acl22_moral.pdf) | [[Data Use Agreement]](https://forms.gle/kp8bPgFoBHeNL4Q68) | [[Data]](https://www.dropbox.com/sh/m46z42nce8x0ttk/AABuSZiA6ESyrJNWmgTPrfuRa?dl=0)

## *What is MIC?* 

Open-domain or "chit-chat" conversational agents often reflect insensitive, hurtful, or contradictory viewpoints that erode a user’s trust in the integrity of the system. Moral integrity is one important pillar for building trust. 

`MIC` is a dataset that can help us understand chatbot behaviors through their latent values and moral statements. `MIC` contains 114k annotations, with 99k distinct "Rules of Thumb" (RoTs) that capture the moral assumptions of 38k chatbot replies to open-ended prompts. These RoTs represent diverse moral viewpoints, with the following distribution of underlying moral foundations: 

* ![51%](https://progress-bar.dev/51) **Care:** wanting someone or something to be safe, healthy, and happy. (58k chatbot replies)
* ![21%](https://progress-bar.dev/21) **Fairness:** wanting to see individuals or groups treated equally or equitably. (24k)
* ![19%](https://progress-bar.dev/19) **Liberty:** wanting people to be free to make their own decisions. (22k)
* ![19%](https://progress-bar.dev/19) **Loyalty:** wanting unity and seeing people keep promises or obligations to an in-group. (22k)
* ![18%](https://progress-bar.dev/18) **Authority:** wanting to respect social roles, duties, privacy, peace, and order. (20k)
* ![11%](https://progress-bar.dev/11) **Sanctity:** wanting people and things to be clean, pure, innocent, and holy. (13k)

We can train encoder-decoder models to automatically generate RoT explanations for chatbot behaviors. This could facilitate explainable downstream applications. For example, we could train RL systems that demote chatbot replies which fall into certain moral classes or train safety classifiers that guide systems towards the desired behaviors, with sensitivity towards ideological and political difference.

In the RoT generation task, we find our best models match the quality, fluency, and relevance of human annotations, but they still generate irrelevant RoTs nearly 28% of the time. This suggests that the proposed generation task is not yet solved and that `MIC` can continue to serve as resource for ongoing work in developing morally-consistent conversational agents.

## *Where can I download the data?*

If you have not already, please complete our short [Data Use Agreement](https://forms.gle/kp8bPgFoBHeNL4Q68). Then follow [this link](https://www.dropbox.com/sh/m46z42nce8x0ttk/AABuSZiA6ESyrJNWmgTPrfuRa?dl=0) to download `MIC.csv`.


## Full Project Pipeline (annotation + experiments + analysis)

### 0. Project Environment
* CUDA, cudnn
* [anaconda](https://www.anaconda.com/products/individual)

1. Prepare separate environment for running GPT-Neo (requires old version of PyTorch) and that for generation
```bash
conda create --name neo python=3.7
conda activate neo
pip install -r requirements_neo.txt
conda deactivate

conda activate summarize
pip install -r requirements_summarize.txt
conda deactivate
```

2. Create main project environment
```bash
conda create --name morals python=3.7
conda activate morals
pip install -r requirements_morals.txt
```

3. Intstall PyTorch (be sure `cudatoolkit` matches your version of CUDA)
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```


### 1. Build a Collection of Moral QA Pairs

##### Download data and resources

Download r/AskReddit Q/A data from the public [Reddit-QA-Corpus](https://github.com/FionnD/Reddit-QA-Corpus) as well as the [Expanded Morality Lexicon](https://databank.illinois.edu/datasets/IDB-3957440) by running

```bash
bash get_data.sh
```

##### Filter questions via lexicons

1. Extract moral foundations keywords from original Q/A pairs
```bash
python find_morals.py --column "questions"
python find_morals.py --column "answers"
```

2. Filter Q/A pairs so that both Q and the Reddit answer have at least one word in the Expanded Morality Lexicon, the Q has a question mark and does not contain the token 'reddit'
```bash
python filter_questions.py
```

##### Use chatbots to answer AskReddit prompts

Generate answers using [BlenderBot](https://huggingface.co/transformers/model_doc/blenderbot.html). You can use the --start_idx and --end_idx flags to run the generations on only a subset of the prompts and parallelize.

```bash
python chatbot_generation/generate_answer.py --model_name blenderbot --indir "data/raw/Reddit-QA-Corpus/filtered/" --outdir "data/prepared/blenderbot_generations/" --use_gpu
```

Generate answers using [DialoGPT](https://huggingface.co/transformers/model_doc/dialogpt.html). You can use the --start_idx and --end_idx flags to run the generations on only a subset of the prompts and parallelize.

```bash
python chatbot_generation/generate_answer.py --model_name dialoGPT --indir "data/raw/Reddit-QA-Corpus/filtered/" --outdir "data/prepared/dialoGPT_generations/" --use_gpu
```

Generate answers using [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-2.7B). You can use the --start_idx and --end_idx flags to run the generations on only a subset of the prompts and parallelize.

```bash
conda activate neo
python generate_answer.py --model_name "gpt-neo" --indir "data/raw/Reddit-QA-Corpus/filtered/" --outdir "data/prepared/gpt-neo_generations/" --use_gpu
```

##### Filter QA pairs via lexicons

1. Find moral words in the chatbot responses

``` bash
python find_morals.py --column "blenderbot_A0" --indir "data/prepared/blenderbot_generations"

python find_morals.py --column "dialoGPT_A0" --indir "data/prepared/dialoGPT_generations"

python find_morals.py --column "gpt-neo_A0" --indir "data/prepared/gpt-neo_generations"
```

2. Filter via lexicons

```bash
python filter_qa.py --column "blenderbot_A0" --indir "data/prepared/blenderbot_generations"

python filter_qa.py --column "dialoGPT_A0" --indir "data/prepared/dialoGPT_generations"

python filter_qa.py --column "gpt-neo_A0" --indir "data/prepared/gpt-neo_generations"
```

##### Run the QAFiltering Task

Prepare input files for HIT using data from `data/prepared/{model}_agg/filtered.csv`, here swapping `{model}` for "blenderbot" or "dialoGPT" or "gpt-neo", adjusting also the start and end flags.

```bash
hit/Task_QAFiltering/prepare_hit.py --model {model} --start_idx 0 --end_idx 100
```

For full output CSVs from the HIT, use the download link [here](https://www.dropbox.com/sh/w0ux1d93ud37r0t/AAAswYLgcD-ZEdJIpPoOA8PNa?dl=0) and place them in `hit/Task_QAFiltering/output/`

##### Train classifiers to filter QA pairs

1. Create train / dev / test splits for each of the chatbot distributions by running

```bash
hit/Task_QAFiltering/make_splits.py --model {model} --output "qa_filtering_clf/splits"
```

2. Tune classifiers via `tuning_script.sh`

```bash
bash tuning_script.sh 0 "sufficient" "qa_filtering_clf/splits/blenderbot" &
bash tuning_script.sh 1 "moral" "qa_filtering_clf/splits/blenderbot" &

bash tuning_script.sh 2 "sufficient" "qa_filtering_clf/splits/dialoGPT" &
bash tuning_script.sh 3 "moral" "qa_filtering_clf/splits/dialoGPT" &

bash tuning_script.sh 4 "sufficient" "qa_filtering_clf/splits/gpt-neo" &
bash tuning_script.sh 5 "moral" "qa_filtering_clf/splits/gpt-neo" &
```

3. Train the best performing model using `qa_filtering_clf/sentence_pairs_clf.py` and run it using `python run_sentence_pairs_clf.py`. For example:

```
qa_filtering_clf/sentence_pairs_clf.py --batchsize 16 --epochs 5 --lr "5e-5" --gpu 0 --label "dialoGPT_0" --input "qa_filtering_clf/splits/dialoGPT" --output "qa_filtering_clf/results_dialoGPT"

python qa_filtering_clf/run_sentence_pairs_clf.py --input "data/prepared/dialoGPT_generations_agg/filtered.csv" --output "qa_filtering_clf/dialoGPT_filtered_moral_predictions.txt" --sent2 "dialoGPT_A0" --path_to_model "qa_filtering_clf/results_dialoGPT/albert-base-v2-moral-16-5-5.pt"
```

##### Use classifier outputs to filter QA pairs

```
qa_filtering_clf/clf_filtering.py --model "blenderbot"
qa_filtering_clf/clf_filtering.py --model "dialoGPT"
qa_filtering_clf/clf_filtering.py --model "gpt-neo"
```

### 2. Run the Main RoT Annotation Task

##### Note: 

To start at this step, download the filtered QAs from one of the following links and place each in a new directory `.data/prepared/{model}_generations_agg/{model}_filtered_clf_sufficient_moral.csv` where `{model}` is replaced with the name of the model (`blenderbot`, `dialoGPT`, `gpt-neo`)

* [Dropbox](https://www.dropbox.com/sh/q7tjmvo301j4gmu/AAAXIMfFIJLJSw_j0LM0ErFca?dl=0)

Prepare hit files (choosing the `start_idx` and `end_idx` to indicate which portions of the )

```bash
python ./hit/Task_QAFiltering/prepare_hit.py --model "blenderbot" --start_idx 0 --end_idx 100 
```

Set up the ***Political Leaning Personality Test*** (after you modify the file with the proper API keys)

```
python ./hit/Qual_PoliticalLeaning/qual_request.py
```

Set up main ***Qualification Test*** (after you modify the file with the proper API keys)

```
python ./hit/Task_MoralEvaluation/qual_request.py
```

Run the `MoralEvaluation.html` task on MTurk, using input from `./hit/Task_QAFiltering/input`, and saving output with the format  `./hit/output/filtered_clf_sufficient_moral_blenderbot_0_100_FINISHED.csv`

You can download all raw batch outputs for this RoT Annotation Task here

* [Dropbox](https://www.dropbox.com/sh/pcntf2vl9wquu2w/AAAkhfPX4MqDuJujmsD0BFDka?dl=0)

After running all annotation batches, prepare MIC

```bash
python ./hit/Task_MoralEvaluation/prepare_MIC.py --output "./data/mic/MIC.csv"
```

### 3. Train RoT Generation Models

##### Note: 

To start at this step, download the MIC Data `MIC.csv` from the following link and place it in a new `./data/mic/` directory.

* [Dropbox](https://www.dropbox.com/sh/m46z42nce8x0ttk/AABuSZiA6ESyrJNWmgTPrfuRa?dl=0)

##### Train the three generative models

```bash
CUDA_VISIBLE_DEVICES=0 python -m rot_generation.generate_rots --input "./data/mic/MIC.csv" --output "./rot_generation/output" --epochs 5 --model_checkpoint "gpt2" --gpu 0 --format_string "Q [answ] A [rot] ~ rot"

CUDA_VISIBLE_DEVICES=1 python -m rot_generation.generate_rots --input "./data/mic/MIC.csv" --output "./rot_generation/output" --epochs 5 --model_checkpoint "t5-small" --gpu 1 --format_string "Q [answ] A [rot] ~ rot"

CUDA_VISIBLE_DEVICES=2 python -m rot_generation.generate_rots --input "./data/mic/MIC.csv" --output "./rot_generation/output" --epochs 5 --model_checkpoint "facebook/bart-large" --gpu 2 --format_string "Q [answ] A [rot] ~ rot"
```

##### Decode the generative models

```bash
bash rot_generation/decode_all.sh "rot_generation/output/*" "./data/mic/MIC.csv" "Q [answ] A [rot] ~ rot" 1 "QA"
```

##### Run baseline retrieval generations

```bash
python rot_generation/retrieve_rots.py --input "./data/mic/MIC.csv" --output "./rot_generation/output/retrieval_Q_A_TARGET_rot/"
```


##### Run social-chem models
First, clone [social-chemistry-101](https://github.com/mbforbes/social-chemistry-101) into the root directory and follow instructions to prepare the [GPT2-XL RoT model](https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/models/gpt2-xl_rot_64_5epochs.tar.gz)

Then run the following code

```bash
python -m rot_generation.generate_rots_decoder --model_directory "social-chemistry-101/output/gpt2-xl_rot_64_5epochs" --input "./data/mic/MIC.csv" --output "rot_generation/output_sc" --format_string "QA [rot] ~ rot" --gpu 3 --top_p 0.9
```

##### Compute all results

```bash
python -m rot_generation.metrics --input "rot_generation/output/*Q_A_TARGET_rot*" --output "all_results.csv"
```

### 4. Run Human Evaluation for RoT Generation

Prepare generations for human evaluation

```bash
python ./hit/Task_GeneratedRoTEvaluation/prepare_hit.py --mic "./data/mic/MIC.csv" --input_glob "./rot_generation/output/*Q_A_TARGET_rot*/test_generation*.csv" --output "./hit/Task_GeneratedRoTEvaluation/input/human_eval_25.csv" --nsamples 25
```

Set up the ***Qualification Test*** (after you modify the file with the proper API keys)

```bash
python ./hit/Task_GeneratedRoTEvaluation/qual_request.py
```

Run the `GeneratedRoTEvaluation.html` task on MTurk, using input from `./hit/Task_GeneratedRoTEvaluation/input` and save file to `./hit/Task_GeneratedRoTEvaluation/output/human_eval.csv`

Print inter-annotator agreements and evaluation results

```bash
python ./hit/Task_GeneratedRoTEvaluation/print_eval_results.py --input "./hit/Task_GeneratedRoTEvaluation/output/human_eval.csv"
```

### 5. Train RoT Attribute Classification Models

##### Generate backtranslations

1. Use different temperatures and different languages: German (de) and Russion (ru)
   **Note:** this will not work unless you first install `fairseq` according to [these instructions](https://github.com/pytorch/fairseq)

```bash
python attribute_clf/backtranslate_to_en.py --temperature 1.2 --k 5 --language "de" --target "rot" --input "data/mic/MIC.csv"
python attribute_clf/backtranslate_to_en.py --temperature 0.7 --k 5 --language "de" --target "rot" --input "data/mic/MIC.csv"
python attribute_clf/backtranslate_to_en.py --temperature 0.9 --k 5 --language "de" --target "rot" --input "data/mic/MIC.csv"
python attribute_clf/backtranslate_to_en.py --temperature 1.2 --k 5 --language "ru" --target "rot" --input "data/mic/MIC.csv"
python attribute_clf/backtranslate_to_en.py --temperature 0.7 --k 5 --language "ru" --target "rot" --input "data/mic/MIC.csv"
python attribute_clf/backtranslate_to_en.py --temperature 0.9 --k 5 --language "ru" --target "rot" --input "data/mic/MIC.csv"
```

2. Join backtranslations

```bash
python attribute_clf/join_backtranslations.py
```

##### Run For Hyperparameter Tunings

1. Run for all learning rates and epochs

```bash
CUDA_VISIBLE_DEVICES=0 bash attribute_clf/tuning_script_ordinal.sh 0 "violation-severity" "./data/mic/MIC.csv" "rot" "bert-base-uncased"
CUDA_VISIBLE_DEVICES=0 bash attribute_clf/tuning_script_ordinal.sh 0 "violation-severity" "./data/mic/MIC.csv" "rot" "albert-base-v2"

CUDA_VISIBLE_DEVICES=1 bash attribute_clf/tuning_script_ordinal.sh 1 "rot-agree-collapsed" "./data/mic/MIC.csv" "rot" "bert-base-uncased"
CUDA_VISIBLE_DEVICES=1 bash attribute_clf/tuning_script_ordinal.sh 1 "rot-agree-collapsed" "./data/mic/MIC.csv" "rot" "albert-base-v2"

CUDA_VISIBLE_DEVICES=2 bash attribute_clf/tuning_script_multilabel.sh 2 "moral-vector" "./data/mic/MIC.csv" "rot" "bert-base-uncased"
CUDA_VISIBLE_DEVICES=2 bash attribute_clf/tuning_script_multilabel.sh 2 "moral-vector" "./data/mic/MIC.csv" "rot" "albert-base-v2"


CUDA_VISIBLE_DEVICES=3 bash attribute_clf/tuning_script.sh 3 "A_agrees" "./data/mic/MIC.csv" "QA" "rot" "albert-base-v2"
CUDA_VISIBLE_DEVICES=3 bash attribute_clf/tuning_script.sh 3 "A_agrees" "./data/mic/MIC.csv" "QA" "rot" "bert-base-uncased"

```

2. Find and run the best model for each task (running scripts are just given as an example, but you should swap the hyperparameters to match the best model output from `find_best_model.py`)

>  Violation Severity

```bash
python attribute_clf/find_best_model.py --dir "attribute_clf/cross_validation_violation-severity/" --metric "MSE" --negate --model "bert-base"

python -m attribute_clf.clf_sentence_pairs_ordinal --batchsize 16 --epochs 2 --lr "5e-5" --gpu 0 --label "violation-severity" --input "./data/mic/MIC.csv" --output "attribute_clf/results_violation-severity" --sent1 "rot" --bert_model "bert-base-uncased" --test
```

> Moral Vector

```bash
python attribute_clf/find_best_model.py --dir "attribute_clf/cross_validation_moral-vector/" --metric "f1_macro" --model "bert-base"

python -m attribute_clf.clf_sentence_pairs_multilabel --batchsize 16 --epochs 8 --lr "5e-5" --gpu 2 --label "moral-vector" --input "./data/mic/MIC.csv" --output "attribute_clf/results_moral-vector" --sent1 "rot" --label_classes 0 1 2 3 4 5 --bert_model "bert-base-uncased" --test
```

> Answer Alignment

```bash
python attribute_clf/find_best_model.py --dir "attribute_clf/cross_validation_A_agrees/" --metric "f1_2" --model "bert-base"

python -m attribute_clf.clf_sentence_pairs --batchsize 8 --epochs 3 --lr "5e-5" --gpu 2 --label "A_agrees" --input "./data/mic/MIC.csv" --output "attribute_clf/results_A_agrees" --sent1 "QA" --sent2 "rot" --bert_model "albert-base-v2" --test
```

### 6. Run Human Benchmark for RoT Classification

Prepare labels for human classification

```bash
python ./hit/Task_SecondaryMoralEvaluation/prepare_human_benchmark.py --input "./data/mic/MIC.csv" --nsamples 300
```

### 7. Compute Inter-annotator agreement for Moral Evaluation

Prepare generations for human evaluation [here `{input}` is the name of a file from the primary MoralEvaluation task (e.g. `filtered_clf_sufficient_moral_blenderbot_0_100_FINISHED.csv`)]

```bash
python ./hit/Task_SecondaryMoralEvaluation/prepare_hit.py --input "{input}"
```

Set up the ***Qualification Test*** (after you modify the file with the proper API keys)

```bash
python ./hit/Task_SecondaryMoralEvaluation/qual_request.py
```

Run the `SecondaryMoralEvaluation.html` task on MTurk, using input from `./hit/Task_SecondaryMoralEvaluation/input` and save files to `./hit/Task_SecondaryMoralEvaluation/output`

Finally, create cleaned output file

```bash
python ./hit/Task_SecondaryMoralEvaluation/prepare_outfile.py --output "./hit/Task_SecondaryMoralEvaluation/secondary_moral_eval_oct23.csv"
```

### 8. Generate plots for the paper

Retrieve the political leanings of all workers, here replacing `{qual_id}` with the appropriate MTurk QualificationTypeId for the `Qual_PoliticalLeaning` task

```bash
python ./hit/Qual_PoliticalLeaning/lookup_worker_leanings.py --qual_id "{qual_id}"
```

Run everything in the jupyter notebook, `dataset-statistics.ipynb`
