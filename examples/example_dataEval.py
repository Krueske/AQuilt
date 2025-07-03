import os
import json
import torch
import numpy as np
import random
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
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
lora_path = sys.argv[2]
judged_data_path = sys.argv[3]
llm = LLM(model=model_path,tensor_parallel_size=1, max_model_len=8196, gpu_memory_utilization=0.9, enable_lora=True)
sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=1024)
print("model prepared")

input_file_name = sys.argv[1]

data = [{
        "context": "In healthy individuals, blood pressure (BP) decreases, or \"dips\", during sleep. Ethnicity and high daytime blood pressure level are known markers of nondipping status. The literature on psychological markers of nondipping is scant but suggests that anger/hostility and chronic stress may be contributors to nondipping.We have investigated this phenomenon in drug-free hypertensives who participated in a clinical trial and supplied extensive demographic, psychological, and biological risk factor data after medication washout prior to any treatment.Sixty-two patients were available for analysis (n = 30 nondippers). While most studies focus only on systolic BP nondipping, we explicitly studied both systolic and diastolic BP dipping as outcomes given that both have prognostic value.Hierarchical multiple regression revealed that predictor variables in total accounted for 38% of variance in systolic blood pressure dipping and 44% of variance in diastolic blood pressure dipping. A significant positive predictor was alcohol consumption (beta = 0.37, t = 2.8, p = 0.007) for systolic BP and beta = 0.43, t = 3.7, p = 0.001 for diastolic BP), and an anger diffusion preference was also a positive predictor (beta = 0.42, t = 2.7, p = 0.01) for systolic BP dipping. No measure of trait negative affect reached significance as a predictor for systolic or diastolic BP dipping.",
        "qa_pair": {
            "question": "In a study of 62 drug-free hypertensive patients, was alcohol consumption identified as a significant positive predictor of both systolic and diastolic blood pressure dipping, and was an anger diffusion preference also identified as a significant positive predictor for systolic blood pressure dipping?Yes, no, or maybe?",
            "thinking_steps": "1. Understand the Question: The question asks whether alcohol consumption and anger diffusion preference were significant positive predictors of systolic and diastolic blood pressure dipping in a study of 62 drug-free hypertensive patients. 2. Analyze the Text: The text states that alcohol consumption was a significant positive predictor for both systolic (beta = 0.37, p = 0.007) and diastolic (beta = 0.43, p = 0.001) blood pressure dipping. Additionally, anger diffusion preference was a significant positive predictor for systolic blood pressure dipping (beta = 0.42, p = 0.01). 3. Logical Reasoning: The text directly supports the statement that both alcohol consumption and anger diffusion preference were significant positive predictors for the specified outcomes. 4. Choose the Best Answer: The text confirms the statement, so the answer is 'Yes'.",
            "answer": "Yes"
        }
        }]

# Process the data from the input
results_dict = []
ag_prompts = []
for i in range(len(data)):
    try:
        qa_pair = "```json\n" + json.dumps(data[i]["qa_pair"], ensure_ascii=False) + "\n```"
    except:
        continue
    results_dict.append(data[i])
    user_prompt1 = user_qe_prompt.format(text=data[i]["context"],qa_pair=qa_pair)
    ag_prompts.append(system_prompt + user_prompt1)


# Generate the quality evaluation using the LLM
all_results = []
outputs = llm.generate(ag_prompts, sampling_params, lora_request=LoRARequest("aquilt_eval", 1, lora_path))
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
    num += 1

# Save the results to a JSON file
with open(judged_data_path, 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=4, ensure_ascii=False)
