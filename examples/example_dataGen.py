import os
import sys
import json
import torch
import numpy as np
import random
import pandas as pd
from vllm import LLM, SamplingParams
import pyarrow.parquet as pq
from tqdm import tqdm
def setup_seed(seed):
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)   
    np.random.seed(seed)   
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed) 

setup_seed(0)
task_prompt_dict ={
"单项选择问答":"""请你从提供的参考资料中生成一个单项选择题帮助学生更好地掌握相关知识：
其中单项选择题题目应该包括一个问题（question），四个选择项（options）标记位A、B、C和D，其中一个是问题的答案(answer);
同时你还需要生成解答问题的思考步骤(thinking_steps)，以及这个问题的答案(answer)。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"多项选择问答":"""请你从提供的参考资料中生成一个多项选择题帮助学生更好地掌握相关知识：
其中多项选择题题目应该包括一个问题（question），多个选择项（options）标记位A、B、C、D、E（以此类推），其中一个或多个是问题的答案(answer);
同时你还需要生成解答问题的思考步骤(thinking_steps)，以及这个问题的答案(answer)。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"闭卷问答":"""请你从提供的参考资料中生成一个不需要参考文本回答的闭卷问答对帮助学生更好地掌握相关知识：
这个问答对应该包括一个问题（question），同时你还需要生成解答问题的思考步骤(thinking_steps)，以及这个问题的答案(answer)。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"开卷问答":"""请你从提供的参考资料中生成一个可以参考用户提供的参考资料进行回答的开卷问答对帮助学生更好地掌握相关知识：
这个问答对应该包括一个问题（question），同时你还需要生成解答问题的思考步骤(thinking_steps)，以及这个问题的答案(answer)。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"single choice question answering": """Please generate a single-choice question from the provided reference materials to help students better grasp the relevant knowledge:
The single-choice question should include a question, four options labeled A, B, C, and D, one of which is the answer to the question;
At the same time, you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"multi choice question answering": """Please generate a multiple-choice question from the references provided to help students better grasp the knowledge:
The multiple-choice question should include a question with multiple options tags A, B, C, D, E (and so on), one or more of which are the answers to the questions;
At the same time, you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"close question answering": """Please generate a closed-book question and answer pair from the provided reference materials that do not require reference text to answer to help students better grasp the relevant knowledge:
This Q&A pair should include a question, and you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"open question answering": """Please generate an open-book Q&A pair from the provided reference materials to help students better grasp the relevant knowledge:
This Q&A pair should include a question, and you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"data complexity evaluation": """Please score the complexity of the user's instruction to help students understand the complexity of the questions.
There are 5 levels of complexity, which are: 1 point (Simple Question), 2 points (Basic Question), 3 points (Moderate Complexity Question), 4 points (Higher Complexity Question), 
You'll first need to analyze the complexity of the question before grading it.
And output in the following JSON format:
```json
{"analysis_steps": "xxx", "score": "xxx"}
```
""",
"data quality evaluation": """Please score the quality of the user's instruction and response to help students understand the quality of the question and response.
There are 5 levels of quality, which are: 1 point - Basic requirements met (Basic Level), 2 points - Basic requirements met with some quality (Qualified Level), \
3 points - Good quality, meeting most requirements (Good Level), 4 points - High quality, meeting all requirements and exceeding expectations (Excellent Level), \
5 points - Excellent quality, exceeding all requirements with professional contributions (Outstanding Level)
You'll first need to analyze the quality of the question and response before grading it.
And output in the following JSON format:
```json
{"analysis_steps": "xxx", "score": "xxx"}
```
""",
"text summarization": """Please generate a concise summary Q&A pairs of the provided text to help students better understand the main points:
The summary should capture the key ideas and essential information from the text.
The content you generate should include a question, and you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"text generation": """Please generate a text-generated Q&A pair based on the text provided to help students learn:
The resulting text should be well-structured and relevant to the given text.
The content you generate should include a question, and you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"natural language inference": """Please generate a logical inference question from the provided reference materials to help students better grasp the relevant knowledge:
Logical inference questions generally ask whether a judgment or piece of knowledge is correct, with answers including "yes, no, maybe" three options.
The content you generate should include a question, and you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"text classification": """Generate a text classification task based on the text provided to help students understand the content of the text:
Classifications should be accurate and relevant to the given text.
The content you generate should include a question, and you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"文本摘要": """请根据提供的文本生成一个合理的摘要问答对，摘要应捕捉文本中的关键思想和基本信息，以帮助学生更好地理解主要观点：
你生成的内容应该包括一个问题，同时你还需要生成解答问题的思考步骤，以及这个问题的答案。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"文本生成": """请根据提供的文本生成一个文本生成问答对，以帮助学生学习相关知识并增强文本生成能力：
你生成的内容应该包括一个问题，同时你还需要生成解答问题的思考步骤，以及这个问题的答案。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"自然语言推断": """请根据提供的参考资料生成一个逻辑推断题，以帮助学生更好地掌握相关知识：
你生成的内容应该包括一个问题，同时你还需要生成解答问题的思考步骤，以及这个问题的答案。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"文本分类": """请根据提供的文本生成一个文本分类任务，以帮助学生理解文本的内容与类别：
分类应准确且与给定的文本相关。
你生成的内容应该包括一个问题，同时你还需要生成解答问题的思考步骤，以及这个问题的答案。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
"""
}
zh_prompt_template = """以下是给定的参考资料：
[参考资料开始]
{text}
[参考资料结束]
请根据要求完成问答对生成。
"""
en_prompt_template = """Here is the provided reference material:
[reference material begin]
{text}
[reference material end]
Please complete the Q&A pair based on the requirements.
"""
model_path = sys.argv[1]
output_data_path = sys.argv[2]

num_gens = 1
llm = LLM(model=model_path,tensor_parallel_size=1,max_model_len=8192)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024, n=num_gens)


unlabeled_data = ["P wave dispersion (PD) is considered to reflect the heterogeneous conduction in atria. We investigated whether there was a correlation between the left ventricular (LV) relaxation and PD.Fifty-three hypertensive patients<or =60 years old were divided into two groups: Group A, 27 patients, aged 54+/-5 years with the impaired LV relaxation and Group B, 26 patients, aged 51+/-8 years with normal LV relaxation. The P wave durations were measured in all 12 leads of ECG and PD was defined as the difference between maximum and minimum P wave duration (Pmax-Pmin). Mitral inflow velocities (E and A), E deceleration time (DT), isovolumic relaxation time (IVRT), left atrial and ventricular diameters, and wall thickness of LV were obtained by echocardiography. Clinical characteristics of both groups were comparable. The wall thickness of LV, Pmax, and left atrial dimension were not different in both groups. A velocity was higher (P<0.001), but E velocity (P=0.03) and E/A ratio (P<0.001) were lower in group A than in group B. IVRT and DT were also significantly longer in group A. PD was significantly higher in group A compared to group B (51+/-9 vs 41+/-11 ms, P=0.01). This difference resulted from the Pmin (61+/-10 vs 67+/-9 ms, P=0.03, respectively). Multivariate analysis revealed a significant correlation between PD and A velocity (r=0.46, P=0.01), E/A ratio (r=-0.53, P=0.001), DT (r=0.65, P<0.001), and IVRT (r=0.73, P<0.001).",
"In healthy individuals, blood pressure (BP) decreases, or \"dips\", during sleep. Ethnicity and high daytime blood pressure level are known markers of nondipping status. The literature on psychological markers of nondipping is scant but suggests that anger/hostility and chronic stress may be contributors to nondipping.We have investigated this phenomenon in drug-free hypertensives who participated in a clinical trial and supplied extensive demographic, psychological, and biological risk factor data after medication washout prior to any treatment.Sixty-two patients were available for analysis (n = 30 nondippers). While most studies focus only on systolic BP nondipping, we explicitly studied both systolic and diastolic BP dipping as outcomes given that both have prognostic value.Hierarchical multiple regression revealed that predictor variables in total accounted for 38% of variance in systolic blood pressure dipping and 44% of variance in diastolic blood pressure dipping. A significant positive predictor was alcohol consumption (beta = 0.37, t = 2.8, p = 0.007) for systolic BP and beta = 0.43, t = 3.7, p = 0.001 for diastolic BP), and an anger diffusion preference was also a positive predictor (beta = 0.42, t = 2.7, p = 0.01) for systolic BP dipping. No measure of trait negative affect reached significance as a predictor for systolic or diastolic BP dipping."]
task_type = "single choice question answering"  # Change this to the desired task type
results = []
texts_with_prompts = []
for i in range(len(unlabeled_data)):
    texts_with_prompts.append(task_prompt_dict[task_type] + zh_prompt_template.format(text=unlabeled_data[i]))

for idx, output in enumerate(tqdm(outputs)):
    data_item = unlabeled_data[idx] # The original data item
    for seq in output.outputs:
        generated_text = seq.text
        try:
            cleaned_text = generated_text.replace("```json", "").replace("```", "").strip()
            qa_pair = json.loads(cleaned_text)
            
            results.append({
                "context": data_item,
                "generated_text": generated_text,
                "task_type": task_type,
                "qa_pair": qa_pair
            })
        except json.JSONDecodeError:
            continue
with open(output_data_path, "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
