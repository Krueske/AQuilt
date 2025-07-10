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
You can use the following script to generate synthetic instruction data from unlabeled text.
```bash
CUDA_VISIBLE_DEVICES=0 python ./dataGen.py \
  --model_path /path/to/AQuilt \
  --eval_lora_path /path/to/AQuilt_eval_lora \
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

Parameter Explanation:

- `--model_path`: Path to AQuilt model

- `--eval_lora_path`: Path to LoRA adapters for Self-Inspection when `--eval=true`

- `--eval`: Enable Self-Inspection mode (true/false)

- `--input_file`: Text file containing raw unlabeled data

- `--output_file`: JSON output file for generated instructions

- `--task_type`: Target task type

- `--task_prefix`: Custom prefix added to all generated instructions

- `--num_gen_per_text`: Number of instructions to generate per input unlabeled data

- `--temperature`: Creativity control

- `--top_p`: Nucleus sampling threshold

- `--seed`:Random seed for reproducibility

📄 Input File Format (input.txt)

Plain text file with one unlabeled text per line:
```txt
Direct pressure applied on the inner ear cannot …… the other group.
We evaluate the outcome vesicoureteral reflux (VUR) in …… respectively (p>0.05).
Radio frequency ablation (RFA) has been used as a …… border of ablation.
To investigate the rates of Staphylococcus aureus carriage …… was documented for 7 (50%).
Lichen aureus is localized variant of persistent pigmented purpuric dermatitis that …… intraepithelial vessels.
```

💾 Output Format (output.json)

JSON file with structured instructions:
```json
[
  {
        "context": "Direct pressure applied on the inner ear cannot …… the other group.",
        "task_type": "natural language inference",
        "qa_pair": {
            "question": "Does direct pressure applied on the inner ear during scuba diving induce hearing loss, based on the findings that perforating the eardrums of guinea pigs before diving resulted in no variation in auditory threshold and cochlear spontaneous activity, even during extreme diving conditions?Yes, no, or maybe?",
            "thinking_steps": "1. Understand the Question: The question asks whether direct pressure on the inner ear during scuba diving causes hearing loss, based on experimental findings with guinea pigs. 2. Analyze the Text: The text describes an experiment where guinea pigs with perforated eardrums were subjected to diving conditions, including extreme depths and durations. No changes in auditory threshold or cochlear spontaneous activity were observed in this group. 3. Logical Reasoning: Since the experiment did not observe any hearing loss or changes in auditory function in the group with perforated eardrums, it suggests that direct pressure on the inner ear does not induce hearing loss. 4. Choose the Best Answer: The text supports the conclusion that direct pressure on the inner ear does not induce hearing loss, as evidenced by the lack of variation in auditory parameters in the experimental group.",
            "answer": "No"
        },
        "analysis_steps": "1. The question is clear and specific, asking whether direct pressure on the inner ear during scuba diving induces hearing loss based on experimental findings. 2. The response is relevant and directly addresses the question by referencing the experimental findings with guinea pigs. 3. The response provides a logical reasoning process, explaining how the lack of observed changes in auditory parameters supports the conclusion that direct pressure does not induce hearing loss. 4. The grammar and expression in the response are fluent and clear, with no spelling errors. 5. The response provides a comprehensive solution by not only answering the question but also explaining the reasoning behind the conclusion, which helps the user understand the context and implications of the findings.",
        "score": 4
    },
]
```

⚠️ Important Notes

- When using eval=true, you must provide eval_lora_path
- The types of tasks you can choose include: `single choice question answering`, `multi choice question answering`, `close question answering`, `open question answering`, `text summarization`, `text generation`, `natural language inference`, `text classification`, `extractive question answering`, `natural language understanding`, as well as their corresponding Chinese versions.
- If you want to generate customized tasks (add `task_prefix`), it is recommended that the task types be adjusted to close question answering or open question answering.

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
