import os
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
import random
import pandas as pd
from vllm import LLM, SamplingParams
import pyarrow.parquet as pq
import sys
def setup_seed(seed):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法

setup_seed(0)

system_prompt = """Please score the quality of the user's instruction and response to help students understand the quality of the question and response based on the provided text.
There are 5 levels of quality, which are: 1 point, 2 points, 3 points, 4 points, 5 points. The higher the score, the better the quality.
You'll first need to analyze the quality of the question and response before grading it.
And output in the following JSON format:
```json
{"analysis_steps": "xxx", "score": "xxx"}
```
"""

user_qe_prompt = """<text begin>
{text}
<text end>
<qa_pair begin>
{qa_pair}
<qa_pair end>"""

model_path = sys.argv[1]
data_path = sys.argv[2]
judged_data_path = sys.argv[3]
llm = LLM(model=model_path,tensor_parallel_size=1, max_model_len=8196, gpu_memory_utilization=0.9)
print("model prepared")

input_file_name = sys.argv[1]

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# data = data[0:10]

results_dict = []
ag_prompts = []
for i in range(len(data)):
    results_dict.append(data[i])
    qa_pair = "```json\n" + json.dumps(data[i]["qa_pair"], ensure_ascii=False) + "\n```"
    user_prompt1 = user_qe_prompt.format(text=data[i]["context"],qa_pair=qa_pair)
    ag_prompts.append(system_prompt + user_prompt1)
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)

all_results = []
outputs = llm.generate(ag_prompts, sampling_params)
num = 0
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    try:
        quality_eval = generated_text.split("```json")[1].split("```")[0].strip()
        quality_eval = json.loads(quality_eval)
        results_dict[num]["analysis_steps"] = quality_eval["analysis_steps"]
        results_dict[num]["score"] = quality_eval["score"]
    except:
        results_dict[num]["analysis_steps"] = f"something wrong:{generated_text}"
        results_dict[num]["score"] = f"something wrong:{generated_text}"
    num+=1
with open(judged_data_path, 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)