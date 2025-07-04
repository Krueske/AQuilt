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

setup_seed(2)

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
"extractive question answering": """Please generate an extractive question answering task based on the provided reference materials to help students better understand the main points:
The content you generate should include a question, and you also need to generate the thinking steps for solving the question, as well as the answer to this question.
And output in the following JSON format:
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"抽取式问答": """请根据提供的参考资料生成一个抽取式问答任务，以帮助学生更好地理解文章内容：
你生成的内容应该包括一个问题，同时你还需要生成解答问题的思考步骤，以及这个问题的答案。
并按照以下json格式输出
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
""",
"自然语言理解": """请根据提供的参考资料生成一个自然语言理解题(例如情感分析，语义分析，实体识别等)，以帮助学生更好地掌握相关知识：
你生成的内容应该包括一个问题，同时你还需要生成解答问题的思考步骤，以及这个问题的答案。
并按照以下json格式输出
```json
{"question": "xxx", "thinking_steps": "xxx", "answer": "xxx"}
```
""",
"natural language understanding": """Please generate a natural language understanding question (such as sentiment analysis, semantic analysis, entity recognition, etc.) based on the provided reference materials to help students better grasp the relevant knowledge:
The content you generate should include a question, and you also need to provide the thinking steps to solve the question, as well as the answer to the question.
Please output in the following JSON format:
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
file_path = sys.argv[2]
output_data_path = sys.argv[3]


llm = LLM(model=model_path,tensor_parallel_size=1,max_model_len=8192,gpu_memory_utilization=0.9)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
results = []
texts_with_prompts = []
for key in data:
    user_prompt1 = en_prompt_template.format(text="".join(data[key]["CONTEXTS"]))
    texts_with_prompts.append(task_prompt_dict["natural language inference"] + user_prompt1)
texts_with_prompts = texts_with_prompts*2000
texts_with_prompts = texts_with_prompts[0:20000]
num_gens = 1
for num_gen in range(num_gens):
    outputs = llm.generate(texts_with_prompts, sampling_params)
    num = -1
    for idx in tqdm(range(len(outputs))):
        num+=1
        output = outputs[idx]
        generated_text = output.outputs[0].text
        try:
            qa_pair = json.loads(generated_text.replace("```json","").replace("```","").strip())
        except:
            continue
        context = output.prompt.split("[reference material begin]")[1].split("[reference material end]")[0].strip()
        results.append({"context":context,"qa_pair":qa_pair})
with open(output_data_path, "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
