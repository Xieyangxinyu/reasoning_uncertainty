# Introduction

This document provides instructions on how to set up the environment and reproduce the results for the paper [Statistical Early Stopping for Reasoning Models](https://arxiv.org/abs/2602.13935).

# Environment Setup
We use Python 3.12 and conda for environment management. The required packages are listed in the `requirements.txt` file. We can install these packages using the following command:
```bash
conda create -n reasoning python=3.12 -y
conda activate reasoning
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory of the project with the following variables:
```bash
HF_HOME=./hf_home
TRANSFORMERS_CACHE=./hf_home/hub
HF_TOKEN=your_huggingface_token_here
```

- `HF_HOME` and `TRANSFORMERS_CACHE`: Specify where to cache Hugging Face models and datasets
- `HF_TOKEN`: Your Hugging Face access token (optional, only needed for gated models or to avoid rate limits). You can create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Additional Setup for Inference

The above is sufficient for all the data analysis included in the paper. If, additionally, you'd like to run our methods on models or datasets not included in the paper, you'd use [vllm](https://github.com/vllm-project/vllm) to make LLM inferences, you can install it by running the following command:
```bash
pip install -U vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

Run the following command to activate the virtual environment:
```bash
conda activate reasoning
```

# Inference

To create reasoning traces for a given dataset and model, run the following command:
```bash
python run.py --model_name \"{model}\" --results_path \"{rdir}/\" --data_name \"{dataset}\"
```
where `model` is the name of the LLM model, `rdir` is the directory where the results will be saved, and `dataset` is the name of the dataset. For example, to create reasoning traces for the `Qwen/Qwen3-8B` model and the `mmlu` dataset, we would run:
```bash
python run.py --model_name "Qwen/Qwen3-8B" --results_path "./results/" --data_name "mmlu"
```
This will create a directory named `Qwen/Qwen3-8B` in the `rdir` directory, and save the reasoning traces for the `mmlu` dataset in this directory: `results/Qwen/Qwen3-8B/mmlu_results.jsonl`.

We support a variety of models and datasets. The full list of supported models [`config.py`](config.py). The supported datasets are implemented in [`data/`](data/). 

# Uncertainty Keyword Extraction

We refer the reader to [`src/detect/readme.md`](src/detect/readme.md) for detailed instructions on how to run the uncertainty keyword extraction code. [`src/detect/`](src/detect/) also include codes for detecting whether a given LLM response abstains from answering a question.

In short, the first step is to run the following command to train the random forest classifier on the training data and create the list of uncertainty keywords:
```bash
python -m src.detect.classifier
```

### Train/Cal/Test Splitting

Interested reader may ask how the training data is subsetted. For all dataset, we call `load_data` with the appropriate arguments to load the data, which will provide the desired subset from all the `results.jsonl` files. Interested reader may refer to [`src/utils.py`](src/utils.py) for more details.

# Statistical Early Stopping:

Both early stopping rules introduced in the paper are implemented in `src/stopping_rules.py`. In our implementation, as we evaluate the early stopping rules post-hoc, where the reasoning traces are already fully generated, our code 

# Baselines

### Prompting Baselines

There are two additional arguments that can be passed to the `run.py` script; this is implemented to append additional instructions to the LLM's context and corresponds to the prompting baselines discussed in the paper:

- `--prompt_intervention`: If this argument is passed, the code will use prompt intervention to guide the model's reasoning process. The prompt template to use for prompt intervention can be specified using the `--prompt_template` argument.
- `--prompt_template`: This argument specifies the prompt template to use for prompt intervention. The default value is `1`, which corresponds to the prompt template `"Answer only if you are confident. Otherwise, say "I am not sure."`. If the value is set to `2`, the prompt template will be `"Please solve these problems with criticism. If the problem is reasonable, please think step by step and put your final answer within boxed. If the problem are unreasonable, highlight these issues clearly in your response and provide a succinct explanation."`.

If you run 
```bash
python run.py --model_name "Qwen/Qwen3-8B" --results_path "./results/" --data_name "mmlu" --prompt_intervention
``` 
this will produce reasoning traces under `results_prompt_intervention/Qwen/Qwen3-8B/mmlu_results.jsonl`.

Similarly, if you run 
```bash
python run.py --model_name "Qwen/Qwen3-8B" --results_path "./results/" --data_name "mmlu" --prompt_intervention --prompt_template 2
```
this will produce reasoning traces under `results_prompt_intervention_criticism/Qwen/Qwen3-8B/mmlu_results.jsonl`.

For the prompting baselines, we need to evaluate their performance on the test set. To do so, we use LLM-based abstention detection to determine whether the LLM's response is an abstention or not. 

The first step is to use the following command to evaluate the performance of the prompting baselines on the test set:
```bash
python -m src.compute_metrics --model_name \"{model}\" --results_path \"{rdir}/\" --data_name \"{dataset}\"
```
This will produce a file named `results/{model}/{dataset}_metrics.json` that contains two fields: `correct_well_posed` and `llm_abstention_ill_posed`. The first field is the number of correct predictions made by the LLM on the well-posed questions, and the second field is the number of abstentions made by the LLM on the ill-posed questions.

Now, we may run the following command to compute the performance of the prompting baselines on the test set:
```bash
python -m src.compute_metrics --model_name \"{model}\" --results_path results_prompt_intervention --data_name \"{dataset}\"
python -m src.compute_metrics --model_name \"{model}\" --results_path results_prompt_intervention_criticism --data_name \"{dataset}\"
```
This will produce two files named `results_prompt_intervention/{model}/{dataset}_metrics.json` and `results_prompt_intervention_criticism/{model}/{dataset}_metrics.json`, respectively, that contain the performance metrics for the prompting baselines on the test set. 

Now, if you run 
```bash
python metrics.py
```
it will produce a table summarizing the performance of the prompting baselines on the test set, by creating three files:
- `results_summary.csv`: This file contains the performance metrics for the original LLM on the test set.
- `results_prompt_intervention_summary.csv`: This file contains the performance metrics for the prompting baseline with the `confident` (1) prompt template on the test set.
- `results_prompt_intervention_criticism_summary.csv`: This file contains the performance metrics for the prompting baseline with the `criticism` (2) prompt template on the test set.

You may modify the code in [`metrics.py`](metrics.py) to include additional baselines or datasets.

In [`tables.ipynb`](tables.ipynb), we will use the following commands to compute the false positive rate (FPR) of confidence and criticism:
```
# the FPR of Confidence is 
results_prompt_intervention_summary['fpr'] = results_summary['avg_correct_well_posed'] - results_prompt_intervention_summary['avg_correct_well_posed']
# the FPR of Criticism is
results_prompt_intervention_criticism_summary['fpr'] = results_prompt_intervention_summary['avg_correct_well_posed'] - results_prompt_intervention_criticism_summary['avg_correct_well_posed']
```

You can find the code to compute the power in [`tables.ipynb`](tables.ipynb).


### DEER- and Entropy-Based Confidence Baselines

We refer the reader to [`src/logits/readme.md`](src/logits/readme.md) for detailed instructions on how to run the DEER- and Entropy-based confidence baselines code. We note that because these methods require access to the model's logits, to run this code, vLLM must be installed, as described in the Environment Setup section.

### Probing Baselines

We refer the reader to [`src/probe/readme.md`](src/probe/readme.md) for detailed instructions on how to run the probing baselines code. 


## Reproducing Results in the Paper

To reproduce the early stopping experiments in the paper, please run
```bash
python -m src.simulator --model_name \"{model}\" --results_path results --data_name \"{dataset}\"
```

You can also add `--alpha {alpha}` to specify the desired alpha level for the early stopping rules (default is `0.05`).

You may also add `--ablation` if you wish to implement the ablation study discussed in the paper.

This will produce a file named `results/{model}/{dataset}_stopping_results.jsonl` that contains the results of the early stopping experiments for the specified model and dataset, for three stopping rules: Length-based, Renewal and Maxwise. For example, we may see the following in such a file:

```json
{"stopping_rule": "Length", "alpha": 0.05, "lazy_interval": "none", "ablation": "none", "early_stopping_rate_well_posed": 4.0, "tokens_saved_list_well_posed": [...], "avg_tokens_saved_ill_posed": 554.93, "avg_percentage_saved_ill_posed": 6.672368753506325, "early_stopping_rate_ill_posed": 22.0, "tokens_saved_list_ill_posed": [...], "soft_upperbound": null, "cal_length_quartiles": [52.0, 134.75, 192.5, 1539.5, 9340.0], "test_length_quartiles": [[70.0, 147.0, 214.0, 4231.25, 12386.0], [70.0, 148.75, 203.5, 1003.5, 6003.0]]}
{"stopping_rule": "Renewal", "alpha": 0.05, "lazy_interval": 250, "ablation": "none", "early_stopping_rate_well_posed": 0.0, "tokens_saved_list_well_posed": [...], "avg_tokens_saved_ill_posed": 1067.6, "avg_percentage_saved_ill_posed": 20.73688858885371, "early_stopping_rate_ill_posed": 28.000000000000004, "tokens_saved_list_ill_posed": [...], "soft_upperbound": 40.0, "cal_length_quartiles": null, "test_length_quartiles": null}
{"stopping_rule": "Maxwise", "alpha": 0.05, "lazy_interval": 250, "ablation": "none", "early_stopping_rate_well_posed": 6.0, "tokens_saved_list_well_posed": [...], "avg_tokens_saved_ill_posed": 1867.06, "avg_percentage_saved_ill_posed": 39.17111157471817, "early_stopping_rate_ill_posed": 50.0, "tokens_saved_list_ill_posed": [...], "soft_upperbound": 40.0, "cal_length_quartiles": null, "test_length_quartiles": null}
```

We explain the fields in the above file below:
- `stopping_rule`: The name of the stopping rule used.
- `alpha`: The alpha level used for the stopping rule.
- `lazy_interval`: The bin size $B$ discussed in the paper. This is the interval at which the stopping rule score is computed and whether to stop is decided. It only applies to Renewal and Maxwise stopping rules.
- `ablation`: Whether to implement the ablation study discussed in the paper. We consider two types of ablations: leave one category of uncertainty keywords out, or adding either "maybe" or "wait" to the uncertainty keywords. If no ablation is used, this field is set to "none". 
- `early_stopping_rate_well_posed`: The early stopping rate (ESR) on the well-posed questions. This is the false positive rate (FPR).
- `tokens_saved_list_well_posed`: A list of the number of tokens saved for each well-posed question.
- `avg_tokens_saved_ill_posed`: The average number of tokens saved on the ill-posed questions.
- `avg_percentage_saved_ill_posed`: The average percentage of tokens saved on the ill-posed questions.
- `early_stopping_rate_ill_posed`: The early stopping rate (ESR) on the ill-posed questions. This is the power.
- `tokens_saved_list_ill_posed`: A list of the number of tokens saved for each ill-posed question.
- `soft_upperbound`: The soft upperbound of the power for a given alpha level, which is the maximum of the specified alpha level and the FPR achieved by this rule. 
- `cal_length_quartiles`: The length quartiles of the calibration set, if Length-based stopping rule is used. 
- `test_length_quartiles`: The length quartiles of the test set, separated into well-posed and ill-posed questions, if Length-based stopping rule is used. The first list corresponds to the ill-posed questions, and the second list corresponds to the well-posed questions. We compute this quantity to show the distribution shift between the calibration and test sets.


Now, if you run 
```bash
python stopping_rule_report.py
```

it will produce a table summarizing the performance of the early stopping rules on the test set, by creating a file named `stopping_rule_results_summary.csv`. Please refer to [`stopping_rule_report.py`](stopping_rule_report.py) for more details on how the summary table is created. 

## Tables and Figures

The code to create the tables and figures in the paper is provided in [`tables.ipynb`](tables.ipynb) and [`tables_cross.ipynb`](tables_cross.ipynb), respectively. Please refer to these notebooks for more details.
