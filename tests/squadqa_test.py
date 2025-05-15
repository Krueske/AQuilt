import jieba
from rouge import Rouge
from vllm import LLM, SamplingParams
import re
import numpy as np
import torch
import random
import os
import json
import string
import pyarrow.parquet as pq
import evaluate
import pandas as pd
import ast
import sys
import pyarrow.ipc as ipc
from datasets import load_from_disk
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(0)

model_path = sys.argv[1]
llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.9)
sampling_params = SamplingParams(temperature=0, max_tokens=1024)


def squadqa_test_wotext(test_data_path, model, sampling_params, output_path):
    print("squadqa_test_wotext")
    test_data = []
    # 读取测试数据
    df = pd.read_csv(test_data_path,header=0)
    lines = df.to_dict(orient='records')

    prompts = []
    reference_answers = []
    model_outputs = []
    for i in range(len(df)):
        answers_json = ast.literal_eval(lines[i]["answers"])
        question = df.loc[i, "question"]
        instruction = ""
        input_1 = f"{question}"
        if "llama" in model_path.lower():
            prompts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}{input_1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        elif "qwen" in model_path.lower():
            prompts.append(f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}{input_1}<|im_end|>\n<|im_start|>assistant\n")
        reference_answers.append(answers_json['text'])
    
    # 生成结果
    outputs = model.generate(prompts, sampling_params)
    for output in outputs:
        try:
            if "aquilt" in model_path:
                output_text = output.outputs[0].text.split("Answer:")[-1].strip()
            else:
                output_text = output.outputs[0].text
        except:
            output_text = output.outputs[0].text
            print(output.outputs[0].text)
        model_outputs.append(output_text.strip().lower())
    

    # 保存输出和参考答案到文件
    result_data = []
    for ref, pred in zip(reference_answers, model_outputs):
        result_data.append({
            "reference": ref,
            "prediction": pred
        })
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result_data, output_file, ensure_ascii=False, indent=4)

    # 计算 f1 分数
    p = [{"id": str(i), "prediction_text": p} for i, p in enumerate(model_outputs)]
    r = [
         {"id": str(i), "answers": {"text": l, "answer_start": [0]}}
         for i, l in enumerate(reference_answers)
        ]
    f1_metrics = evaluate.load("./evaluate-main/metrics/squad")
    result = f1_metrics.compute(references=r, predictions=p)
    print("squad score:", result)



filename = model_path.split("/")[-1]
test_data_path = "./SquadQA/test/squad_test.csv"
squadqa_test_wotext(test_data_path, llm, sampling_params, f"./SquadQA/results/{filename}.json")

