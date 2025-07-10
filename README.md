## AQuilt

Overview of the proposed AQuilt framework. The left side illustrates the training process of our data synthesis model, while the right side demonstrates how the trained model automatically synthesizes high-quality domain-specific data.

<img src="images/main_figure.png" alt="Alt Text" style="zoom:50%;" width = "2000" />

## 🔗Links

You can download the AQuilt model from this link: [AQuilt](https://huggingface.co/xiapk7/AQuilt)

## ⚙️Environment Setup

To install all the relevant packages, run the following:

```bash
conda create -n [environment_name] --file requirements.txt
conda activate [environment_name]
```
## 🚀Basic Usage
Instruction Data Generation Script

This script generates synthetic instruction data from unlabeled text using AQuilt. Below is a guide to using the data generation pipeline:
```bash
CUDA_VISIBLE_DEVICES=0 python ./example_dataGen.py \
  --model_path /data/kxp/AQuilt_v0411 \
  --eval_lora_path /data/kxp/LLaMA-Factory/saves/AQuilt_v0411_with_eval/lora/sft \
  --eval true \
  --input_file input.txt \
  --output_file output.json \
  --task_type "natural language inference" \
  --task_predix "" \
  --num_gen_per_text 1 \
  --temperature 0.7 \
  --top_p 0.95 \
  --seed 42
```

⚙️ Full Parameter List

Parameter Required Default Description

--model_path Yes - Path to base model directory<br>(e.g., /data/kxp/AQuilt_v0411)

--eval_lora_path Conditional - Path to LoRA adapters for evaluation<br>Required when eval=true

--eval Yes - Enable evaluation mode (true/false)

--input_file Yes - Text file containing raw input sentences

--output_file Yes - JSON output file for generated instructions

--task_type Yes - Target task type<br>(e.g., "natural language inference", "question answering")

--task_prefix No Empty Custom prefix added to all generated instructions

--num_gen_per_text No 1 Instructions to generate per input line

--temperature No 0.7 Creativity control (0.0-1.0)<br>Lower = more deterministic

--top_p No 0.95 Nucleus sampling threshold (0.0-1.0)

--seed No 42 Random seed for reproducibility

📄 Input File Format (input.txt)

Plain text file with one unprocessed sentence per line:

A man is walking his dog in the park.
Scientists discovered new marine species near hydrothermal vents.
Electric vehicles require different infrastructure than combustion engines.
Cloud computing enables remote data processing through virtualized resources.


💾 Output Format (output.json)

Generates JSON file with structured instructions:
[
  {
  
  },
  ...
]


⚠️ Important Notes

- 1. When using eval=true, you must provide eval_lora_path
- 2. The types of tasks you can choose include: `single choice question answering`, `multi choice question answering`, `close question answering`, `open question answering`, `text summarization`, `text generation`, `natural language inference`, `text classification`, `extractive question answering`, `natural language understanding`, as well as their corresponding Chinese versions.
- 3. If you want to generate customized tasks (add `task_prefix`), it is recommended that the task types be adjusted to close question answering or open question answering.

## 👨‍💻Experiment

### Data Synthesis

Use the `scripts/dataGen.sh` script to synthesis data, run the following script:

```bash
domain_task=""
model_path=""
unlabeled_data_path=""
output_data_path=""
CUDA_VISIBLE_DEVICES=0 python -u ../dataGen/$domain_task\_sft_dataGen.py $model_path $unlabeled_data_path $outoput_data_path > ./logs/data_gen.log 2>&1 &
```

Options:

- `domain_task`:The type of domain task for which you need synthetic data. , your choices include ("ceval","pubmedqa","squadqa","translation","openend").
- `model_path`:The location of the AQuilt model.
- `unlabeled_data_path`:To use AQuilt for generating synthetic data based on unlabeled documents, you need to specify the location of the unlabeled documents in the corresponding domain.
- `output_data_path`:The storage location for synthetic data.

### Synthetic Data Inspection

Use the `scripts/dataEval.sh` script to evaluate data, run the following script:

```bash
model_path=""
eval_lora_path=""
data_path=""
output_datapath=""
CUDA_VISIBLE_DEVICES=0 python -u data_eval.py $model_path $eval_lora_path $data_path $output_datapath > ./logs/data_eval.log 2>&1 &
```

Options:

- `model_path`:The location of the AQuilt model.
- `eval_lora_path`: Self-inspection-lora can be used for evaluating data quality.
- `data_path`:The location of the synthetic data obtained in the previous step.
- `output_datapath`:The location of the dataset with evaluation results.

Then use `scripts/process_data.sh` to filter the synthetic data:

```bash
judged_data_path=""
train_data_path=""
CUDA_VISIBLE_DEVICES=0 python -u process_data_judged.py $judged_data_path $train_data_path > ./logs/process_data.log 2>&1 &
```

Options:

- `judged_data_path`:The location of the dataset with evaluation results obtained in the previous step.
- `train_data_path`:The location of the dataset processed for training.

### Training

Use the `scripts/train_llama3.sh` to train the downstream model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train LLaMA-Factory/llama3_lora_sft1.yaml

wait

modelPath=Meta-Llama3-8B-Instruct

adapterModelPath=lora_path

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli export \
  --model_name_or_path $modelPath \
  --adapter_name_or_path $adapterModelPath \
  --template empty \
  --finetuning_type lora \
  --export_dir output_model_path \
  --export_size 2 \
  --export_legacy_format False
```

Notes:

- You need to first install Llama-Factory and specify the location of the dataset obtained in the previous step in the `data_info.json` file of Llama-Factory. An example is as follows:

```json
{
  "dataset_name":{
    "file_name":"file_path",
    "columns":{
      "prompt":"question",
      "response":"answer"
    }
  },
}
```

### Evaluation

Use the `scripts/eval.sh` to evaluate the model on target datasets:

```bash
model_path=""
domain_task=""
CUDA_VISIBLE_DEVICES=0 python -u ./tests/$domain_task\_test.py $model_path > ./logs/$domain_task\_eval.log 2>&1 &
```

Options:

- `model_path`: The location of the models that need to be evaluated.
- `domain_task`: The type of domain task that needs to be evaluated， your choices include ("ceval","pubmedqa","squadqa","translation","openend").

Notes:

- For the CEVAL Task, you need to upload the output results to the [CEVAL official website](https://cevalbenchmark.com/index.html) to obtain the test scores, and calculate the average score of the 8 subjects we evaluated (covering 6 different domains).
