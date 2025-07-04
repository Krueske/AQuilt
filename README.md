## AQuilt

Overview of the proposed AQuilt framework. The left side illustrates the training process of our data synthesis model, while the right side demonstrates how the trained model automatically synthesizes high-quality domain-specific data.

<img src="images/main_figure.png" alt="Alt Text" style="zoom:50%;" width = "2000" />

## AQuilt Download

You can download the AQuilt model after anonymity period from this link:[link]

## Environment Setup

To install all the relevant packages, run the following:

```bash
conda create -n [environment_name] --file requirements.txt
conda activate [environment_name]
```
## Quick Start
You can directly refer to the `example_dataGen.py` file and the `example_dataEval.py` file for data synthesis and data evaluation:
```bash
python -u examples/example_dataGen.py "AQuilt_path" "exampleGen.json"
python -u examples/example_dataEval.py "AQuilt_path" "AQuilt_Eval_lora_path" "exampleEval.json"
```
Notes:
- The types of tasks you can choose include: `single choice question answering`, `multi choice question answering`, `close question answering`, `open question answering`, `text summarization`, `text generation`, `natural language inference`, `text classification`, `extractive question answering`, `natural language understanding`, as well as their corresponding Chinese versions.


If you want to generate customized tasks, you can refer to the `example_dataGen_customTask.py` file for data synthesis.
```bash
python -u examples/example_dataGen.py "AQuilt_path" "exampleGen_customTask.json"
```
Notes:
- Please note that the task types should be adjusted to: close question answering(闭卷问答) or open question answering(开卷问答)

## Experiment

## Data Synthesis

Use the `dataGen.sh` script to synthesis data, run the following script:

```bash
domain_task=""
model_path=""
unlabeled_data_path=""
output_data_path=""
CUDA_VISIBLE_DEVICES=0 python -u ./dataGen/$domain_task\_sft_dataGen.py $model_path $unlabeled_data_path $outoput_data_path > ./logs/data_gen.log 2>&1 &
```

Options:

- `domain_task`:The type of domain task for which you need synthetic data. , your choices include ("ceval","pubmedqa","squadqa","translation","openend")
- `model_path`:The location of the AQuilt model
- `unlabeled_data_path`:To use AQuilt for generating synthetic data based on unlabeled documents, you need to specify the location of the unlabeled documents in the corresponding domain.
- `output_data_path`:The storage location for synthetic data.

## Synthetic Data Inspection

Use the `dataEval.sh` script to evaluate data, run the following script:

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

Then use `process_data.sh` to filter the synthetic data:

```bash
judged_data_path=""
train_data_path=""
CUDA_VISIBLE_DEVICES=0 python -u process_data_judged.py $judged_data_path $train_data_path > ./logs/process_data.log 2>&1 &
```

Options:

- `judged_data_path`:The location of the dataset with evaluation results obtained in the previous step.
- `train_data_path`:The location of the dataset processed for training.

## Training

Use the `train_llama3.sh` to train the downstream model:

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

## Evaluation

Use the `eval.sh` to evaluate the model on target datasets:

```bash
model_path=""
domain_task=""
CUDA_VISIBLE_DEVICES=0 python -u ./tests/$domain_task\_test.py $model_path > ./logs/$domain_task\_eval.log 2>&1 &
```

Options:

- `model_path`: The location of the models that need to be evaluated.
- `domain_task`: The type of domain task that needs to be evaluated， your choices include ("ceval","pubmedqa","squadqa","translation","openend").
