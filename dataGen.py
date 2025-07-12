import os
import sys
import json
import torch
import numpy as np
import random
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
import re
def setup_seed(seed):
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)   
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)   
    torch.cuda.manual_seed_all(seed) 

setup_seed(0)
# system prompt for different task types
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

task_type_dict = {
    "单项选择问答": "single choice question answering",
    "多项选择问答": "multi choice question answering",
    "闭卷问答": "close question answering",
    "开卷问答": "open question answering",
    "文本摘要": "text summarization",
    "文本生成": "text generation",
    "自然语言推断": "natural language inference",
    "文本分类": "text classification",
    "抽取式问答": "extractive question answering",
    "自然语言理解": "natural language understanding",
    "single choice question answering": "单项选择问答",
    "multi choice question answering": "多项选择问答",
    "close question answering": "闭卷问答",
    "open question answering": "开卷问答",
    "text summarization": "文本摘要",
    "text generation": "文本生成",
    "natural language inference": "自然语言推断",
    "text classification": "文本分类",
    "extractive question answering": "抽取式问答",
    "natural language understanding": "自然语言理解"
}

# user prompt templates for different languages
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

# System prompt for self-inspection
system_prompt = """Please score the quality of the user's instruction and response to help students understand the quality of the question and response based on the provided text.
There are 5 levels of quality, which are: 1 point, 2 points, 3 points, 4 points, 5 points. The higher the score, the better the quality.
You'll first need to analyze the quality of the question and response before grading it.
And output in the following JSON format:
```json
{"analysis_steps": "xxx", "score": "xxx"}
```
"""
# User prompt for self-inspection
user_qe_prompt = """<text begin>
{text}
<text end>
<qa_pair begin>
{qa_pair}
<qa_pair end>"""


def load_unlabeled_data(input_file):
    """从文本文件加载无标签数据，每行作为独立文本段"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Data synthesis model')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--eval_lora_path', default=None, help='Path to the LoRA model for self-inspection')
    parser.add_argument('--input_file', required=True, help='Path to input text file (one text per line)')
    parser.add_argument('--output_file', required=True, help='Path to output JSON file')
    parser.add_argument('--task_type', choices=['单项选择问答', '多项选择问答', '闭卷问答', '开卷问答', 'single choice question answering', 'multi choice question answering',
                                                'close question answering', 'open question answering', 'text summarization', 'text generation', 'natural language inference',
                                                'text classification', '文本摘要', '文本生成', '自然语言推断', '文本分类', '抽取式问答', 'extractive question answering',
                                                '自然语言理解', 'natural language understanding'], default='close question answering', help='Task type')
    parser.add_argument('--language', type=str, default="en", help='Task language')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--eval', type=bool, default=True, help='Whether to perform self-inspection')
    parser.add_argument('--task_predix', type=str, default="", help='Task prefix for the task type')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
    parser.add_argument('--num_gen_per_text', type=int, default=1, help='Number of qa_pairs to generate per text')
    return parser.parse_args()

def is_chinese_or_english(text):
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    english_chars = re.findall(r'[a-zA-Z]', text)

    if len(chinese_chars) > len(english_chars):
        return "zh"
    else:
        return "en"

def main():
    args = parse_arguments()
    
    setup_seed(args.seed)
    
    try:
        unlabeled_data = load_unlabeled_data(args.input_file)
        print(f"Successfully loaded {len(unlabeled_data)} text entries")
    except Exception as e:
        print(f"unlabeled data load error: {str(e)}")
        sys.exit(1)

    # init model
    if args.eval_lora_path is not None and args.eval:
        llm = LLM(model=args.model_path, tensor_parallel_size=1, max_model_len=8192, enable_lora=args.eval_lora_path is not None and args.eval, max_lora_rank=64)
    else:
        llm = LLM(model=args.model_path, tensor_parallel_size=1, max_model_len=8192, enable_lora=False)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=1024, n=args.num_gen_per_text)
    
    # choose prompt template based on task type
    task_type = args.task_type
    task_type_language = is_chinese_or_english(task_type)
    if task_type_language == "en" and args.language == "zh":
        task_type = task_type_dict[task_type]
    prompt_template = zh_prompt_template if args.language == 'zh' else en_prompt_template
    task_prompt = task_prompt_dict[task_type]
    if args.task_predix:
        task_define = "```json\n{\"question\": \"" + args.task_predix
    else:
        task_define = ""
    # change texts to prompts
    texts_with_prompts = [
        task_prompt + prompt_template.format(text=text) + task_define
        for text in unlabeled_data
    ]
    
    # generate QA pairs
    results = []
    for i, text in enumerate(tqdm(texts_with_prompts, desc="qa pair generation")):
        outputs = llm.generate([text], sampling_params)
        for seq in outputs[0].outputs:
            try:
                generated_text = task_define + seq.text
                cleaned = generated_text.replace("```json", "").replace("```", "").strip()
                qa_pair = json.loads(cleaned)
                
                results.append({
                    "context": unlabeled_data[i],
                    "task_type": args.task_type,
                    "qa_pair": qa_pair
                })
            except json.JSONDecodeError:
                print(f"JSON parsing failed: {seq.text[:100]}...")
    print(f"Generated {len(results)} qa_pairs")
    if args.eval_lora_path is not None and args.eval:
        print("Start self-inspection")
        ag_prompts = []
        for i in range(len(results)):
            qa_pair = "```json\n" + json.dumps(results[i]["qa_pair"], ensure_ascii=False) + "\n```"
            user_prompt1 = user_qe_prompt.format(text=results[i]["context"], qa_pair=qa_pair)
            ag_prompts.append(system_prompt + user_prompt1)
        # generate self-inspection results
        outputs = llm.generate(ag_prompts, sampling_params, lora_request=LoRARequest("aquilt_eval", 1, args.eval_lora_path))
        num = 0
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            try:
                quality_eval = generated_text.split("```json")[1].split("```")[0].strip()
                quality_eval = json.loads(quality_eval)
                results[num]["analysis_steps"] = quality_eval["analysis_steps"]
                results[num]["score"] = quality_eval["score"]
            except json.JSONDecodeError:
                results[num]["analysis_steps"] = f"something wrong:{generated_text}"
                results[num]["score"] = f"something wrong:{generated_text}"
            num += 1
        print(f"Self-inspection completed, length of results: {len(results)}")
    # save results to JSON file
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results has been saved to {args.output_file}")

if __name__ == "__main__":
    main()
