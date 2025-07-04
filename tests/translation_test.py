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
import sys
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

def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def calculate_rouge_l(reference_answers, model_outputs):

    rouge = Rouge()
    scores = []

    for ref, pred in zip(reference_answers, model_outputs):

        ref = " ".join(list(jieba.cut(normalize_zh_answer(ref), cut_all=False)))
        pred = " ".join(list(jieba.cut(normalize_zh_answer(pred), cut_all=False)))

        try:
            score = rouge.get_scores([pred], [ref], avg=True)["rouge-l"]["f"]
        except:
            score = 0.0
            print("Error calculating ROUGE-L score")
            print("Reference:", ref)
            print("Prediction:", pred)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def translation_test(test_data_path, model, sampling_params, output_file):
    print("translation_test")
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        temps = f.readlines()
        for temp in temps:
            test_data.append(json.loads(temp))
    
    prompts = []
    reference_answers = []
    model_outputs = []
    
    for data in test_data:
        instruction = data["instruction"]
        input_1 = data["input"]
        answer = data["answer"]
        
        if "llama" in model_path.lower():
            prompts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}{input_1}\n请直接输出翻译结果。<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        elif "qwen" in model_path.lower():
            prompts.append(f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}{input_1}<|im_end|>\n<|im_start|>assistant\n")
        reference_answers.append(answer)

    outputs = model.generate(prompts, sampling_params)
    for output in outputs:
        try:
            output_text = output.outputs[0].text.split("Answer:")[-1].strip()
        except:
            output_text = output.outputs[0].text
        model_outputs.append(output_text)
    
    result_data = []
    for ref, pred in zip(reference_answers, model_outputs):
        result_data.append({
            "reference": ref,
            "prediction": pred
        })
    with open(output_file, "w", encoding="utf-8") as output_file:
        json.dump(result_data, output_file, ensure_ascii=False, indent=4)
    rouge_l = calculate_rouge_l(reference_answers, model_outputs)
    print(f"ROUGE-L score: {rouge_l}")

    return rouge_l


filename = model_path.split("/")[-1]
translation_test("./data/translation/test/5_3.json", llm, sampling_params, f"./data/translation/results/{filename}.json")
