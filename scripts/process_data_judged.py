import os
import json
import re
import random
import sys
file_path = sys.argv[1]
output_path = sys.argv[2]
results = []
results_score1 = []
results_score2 = []
results_score3 = []
results_score4 = []
results_score5 = []
if "judged" in file_path:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i in range(len(data)):
        try:
            score =  int(data[i]["score"])
        except:
            continue
        try:
            question = data[i]["qa_pair"]["question"]
            if "openend" in file_path.lower():
                instruction = "请分析以下论述题，详细阐述你的观点并可以引用法律条文和相关法律原则。确保你针对每个问题提供充分的论据和分析，以清晰展示你对法律问题的深刻理解和灵活应用能力:"
                question = instruction.replace(":","。") + "\n材料：" + data[i]["context"] + "\n问题：" + question.replace(instruction, "")
            elif "translation" in file_path.lower():
                question = question.replace(",并直接输出翻译结果","")
            elif "pubmed" in file_path.lower():
                if "maybe" in data[i]["qa_pair"]["answer"].lower():
                    continue
                question = "Context: {context}\nBased on the context above, please answer the following question:{question}".format(context=data[i]["context"], question=question)
            if score >= 5:
                results_score5.append({"question":question,"answer":"""Thinking steps:{think_step}\nAnswer:{answer}""".format(
                think_step=data[i]["qa_pair"]["thinking_steps"], answer=data[i]["qa_pair"]["answer"])})
            elif score == 2:
                results_score2.append({"question":question,"answer":"""Thinking steps:{think_step}\nAnswer:{answer}""".format(
                think_step=data[i]["qa_pair"]["thinking_steps"], answer=data[i]["qa_pair"]["answer"])})
            elif score == 3:
                results_score3.append({"question":question,"answer":"""Thinking steps:{think_step}\nAnswer:{answer}""".format(
                think_step=data[i]["qa_pair"]["thinking_steps"], answer=data[i]["qa_pair"]["answer"])})
            elif score == 4:
                results_score4.append({"question":question,"answer":"""Thinking steps:{think_step}\nAnswer:{answer}""".format(
                think_step=data[i]["qa_pair"]["thinking_steps"], answer=data[i]["qa_pair"]["answer"])})
            else:
                results_score1.append({"question":question,"answer":"""Thinking steps:{think_step}\nAnswer:{answer}""".format(
                think_step=data[i]["qa_pair"]["thinking_steps"], answer=data[i]["qa_pair"]["answer"])})
        except:
            continue
    results = results_score1 + results_score2 + results_score3 + results_score4 + results_score5
    if (len(results_score2) + len(results_score1)) > 0.2*len(results):
        high_quality = results_score2 + results_score3 + results_score4 + results_score5
    else:
        high_quality = results_score3 + results_score4 + results_score5
    random.shuffle(high_quality)
    print(len(high_quality))
    filtered_high_quality = []
    if "openend" in file_path.lower() or "translation" in file_path.lower():
        for i in range(len(high_quality)):
            if len(high_quality[i]["question"] + high_quality[i]["answer"]) > 2300:
                continue
            else:
                filtered_high_quality.append(high_quality[i])
    else:
        filtered_high_quality = high_quality
    print(len(filtered_high_quality))
    data_num = len(filtered_high_quality)
    low_quality = []
    for score_id in range(5):
        results_temp = globals()["results_score"+str(score_id+1)]
        if len(low_quality) + len(results_temp) > data_num:
            low_quality = low_quality + random.sample(results_temp, data_num-len(low_quality))
            break
        else:
            low_quality = low_quality + results_temp
    random.shuffle(low_quality)
    print(len(results))
    print(len(filtered_high_quality))
    random_quality = random.sample(results, len(filtered_high_quality))
    random.shuffle(random_quality)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_high_quality, f, ensure_ascii=False, indent=4)
else:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i in range(len(data)):
        try:
            question = data[i]["qa_pair"]["question"]
            if "openend" in file_path.lower():
                instruction = "请分析以下论述题，详细阐述你的观点并可以引用法律条文和相关法律原则。确保你针对每个问题提供充分的论据和分析，以清晰展示你对法律问题的深刻理解和灵活应用能力."
                question = instruction.replace(":","。") + "\n材料：" + data[i]["context"] + "\n问题：" + question.replace(instruction, "")
            elif "translation" in file_path.lower():
                question = question.replace(",并直接输出翻译结果","")
            elif "pubmed" in file_path.lower():
                if "maybe" in data[i]["qa_pair"]["answer"].lower():
                    continue
                question = "Context: {context}\nBased on the context above, please answer the following question:{question}".format(context=data[i]["context"], question=question)
            results.append({"question":question,"answer":"""Thinking steps:{think_step}\nAnswer:{answer}""".format(
            think_step=data[i]["qa_pair"]["thinking_steps"], answer=data[i]["qa_pair"]["answer"])})
        except:
            continue
    print(len(results))
    random.shuffle(results)
    filtered_results = []
    if "openend" in file_path.lower() or "translation" in file_path.lower():
        for i in range(len(results)):
            if len(results[i]["question"] + results[i]["answer"]) > 2300:
                continue
            else:
                filtered_results.append(results[i])
    else:
        filtered_results = results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=4)
    
