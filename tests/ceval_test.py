import os
import sys
import json
import re
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import LlamaTokenizer, AutoConfig, LlamaForCausalLM, GenerationConfig
# vllm 相关导入
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
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



# 加载 vllm 模型
model_path = sys.argv[1]
llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.9)
sampling_params = SamplingParams(temperature=0, max_tokens=1024)
def extract_answer(line, gen_ans):
    choices = ["A", "B", "C", "D"]
    m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
    if len(m) > 0 and m[-1] in choices:
        return m[-1], True
    answer_patterns = [
        r'([ABCD])是正确的',
        r'选项([ABCD])正确',
        r'答案为([ABCD])',
        r'答案是([ABCD])',
        r'答案([ABCD])',
        r'选择([ABCD])',
        r'答案：([ABCD])',
        r'选择答案([ABCD])',
        r'答案应该是([ABCD])',
        r'答案是选项([ABCD])',
        r'答案应该是选项([ABCD])',
        r'答案为选项([ABCD])',
        r'answer is ([ABCD])',
        r'answer: ([ABCD])',
        r'So option ([ABCD])',
        r'So, option ([ABCD])',
        r'answer is:\n\n([ABCD])',
        r'Therefore, option ([ABCD])',
        r'答案是 ([ABCD])'
    ]
    for answer_pattern in answer_patterns:
        m = re.search(answer_pattern, gen_ans, re.M)
        if m:
            answer = m.group(1)
            return answer, False
    m = re.findall(r'[ABCD]', gen_ans, re.M)
    if len(m) >= 1:
        answer = m[0]
        return answer, False
    choices_dict = {}
    pattern = ""
    for c in choices:
        choices_dict[str(line[f'{c}'])] = c
        pattern += re.escape(str(line[f'{c}'])) + "|"
    pattern = pattern[:-1]
    m = re.findall(pattern, gen_ans, re.M)
    if len(m) >= 1:
        answer = choices_dict[m[0]]
        return answer, False
    return random.choice('ABCD'), False

def files_name(directory):
    files_and_folders = os.listdir(directory)
    # 过滤出文件名，假设你只想要文件名，不包括文件夹
    file_names = [file for file in files_and_folders if os.path.isfile(os.path.join(directory, file))]
    # 文件名列表
    subjects = []
    for file_name in file_names:
        subject = re.sub('_test.csv', '', str(file_name))
        subjects.append(subject)
    return subjects

def test(subject_name):
    df = pd.read_csv("./CEVAL/test/{}_test.csv".format(subject_name), header=0)
    lines = df.to_dict(orient='records')
    in_context = ""
    df_dev = pd.read_csv("./CEVAL/dev/{}_dev.csv".format(subject_name), header=0)
    for i in range(len(df_dev)):
        in_context += "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是什么？{answer}\n".format(
            question=df_dev.loc[i, 'question'], A=df_dev.loc[i, 'A'], B=df_dev.loc[i, 'B'], C=df_dev.loc[i, 'C'],
            D=df_dev.loc[i, 'D'], answer=df_dev.loc[i, 'answer'])
    model_type = "instruct"
    if model_type == "instruct":
        in_context = ""
        data_with_prompts = []
        for i in range(len(df)):
            if "llama3" in model_path:
                data_with_prompts.append(
                    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n{in_context}{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是什么？<|eot_id|><|start_header_id|>assistant<|end_header_id|>""".format(
                        subject=subject_name, in_context=in_context, question=df.loc[i, 'question'], A=df.loc[i, 'A'],
                        B=df.loc[i, 'B'], C=df.loc[i, 'C'], D=df.loc[i, 'D']))
            elif "qwen2" in model_path:
                data_with_prompts.append(
                    """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n{in_context}{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是什么？<|im_end|>\n<|im_start|>assistant\n""".format(
                        subject=subject_name, in_context=in_context, question=df.loc[i, 'question'], A=df.loc[i, 'A'],
                        B=df.loc[i, 'B'], C=df.loc[i, 'C'], D=df.loc[i, 'D']))
    elif model_type == "base":
        data_with_prompts = []
        for i in range(len(df)):
            data_with_prompts.append(
            """以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n{in_context}{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是什么？""".format(
                subject=subject_name,in_context=in_context,question=df.loc[i, 'question'], A=df.loc[i, 'A'], B=df.loc[i, 'B'], C=df.loc[i, 'C'], D=df.loc[i, 'D']))
    
    outputs = llm.generate(data_with_prompts, sampling_params=sampling_params)
    subject_res = {}
    for idx in tqdm(range(len(outputs))):
        output = outputs[idx]
        generated_text = output.outputs[0].text
        generated_text1 = generated_text.split("Answer")[-1]
        ans, direct_extract = extract_answer(lines[idx], generated_text1)
        subject_res[str(idx)] = ans
        if idx == 0:
            print(generated_text)
            print(ans)
            print("--------------------------------------------------")
    return subject_res

subjects = files_name('./CEVAL/test')
results = {}
for i in range(len(subjects)):
    subject_name = subjects[i]
    result1 = test(subject_name)
    results[subject_name] = result1
model_name = model_path.split("/")[-1]
with open(f"./CEVAL/results/{model_name}.json", "w", encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
    print("测试完成...")