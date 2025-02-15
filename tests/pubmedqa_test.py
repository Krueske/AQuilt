import json
from vllm import LLM, SamplingParams
import re
# 加载 vllm 模型
model_path = sys.argv[1]
llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.9)
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=512)

# Function to evaluate the model on the PubMedQA dataset
def evaluate_model(model, dataset, ground_truth):
    results = []
    data_with_prompts = []
    for item in dataset:
        pmid = item['pmid']
        question = item['question']
        context = item['context']
        input_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nContext:\n{context}\nBased on the context above, please answer the following question:\n{question}Yes, no or maybe?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        data_with_prompts.append((pmid, input_text))
    
    # Generate the answers using the model
    pmids, prompts = zip(*data_with_prompts)
    outputs = model.generate(prompts, sampling_params=sampling_params)
    
    for idx, output in enumerate(outputs):
        pmid = pmids[idx]
        generated_text = output.outputs[0].text
        answer = generated_text
        
        results.append({
            'pmid': pmid,
            'question': dataset[idx]['question'],
            'context': dataset[idx]['context'],
            'generated_answer': answer,
            'true_answer': ground_truth[pmid]
        })
    
    return results

def judge_answer(generated_answer, true_answer):
    match = re.search(r'\b(yes|no|maybe)\b', generated_answer, re.IGNORECASE)
    if match:
        extracted_answer = match.group(1).lower()
        return extracted_answer == true_answer.lower()
    return False
    

# 计算准确率
def calculate_accuracy(results):
    correct = 0
    for result in results:
        if judge_answer(result['generated_answer'], result['true_answer']):
            correct += 1
    return correct / len(results)

# 加载数据集
with open('./PubMedQA/test/test_ground_truth.json', 'r', encoding="utf-8") as f:
    test_ground_truth = json.load(f)

with open('./PubMedQA/test/ori_pqal.json', 'r', encoding="utf-8") as f:
    ori_pqal = json.load(f)

# 构建测试数据集
test_dataset = []
for pmid, label in test_ground_truth.items():
    if pmid in ori_pqal:
        test_dataset.append({
            'pmid': pmid,
            'question': ori_pqal[pmid]['QUESTION'],
            'context': "".join(ori_pqal[pmid]['CONTEXTS'])
        })

# 评估模型
results = evaluate_model(llm, test_dataset, test_ground_truth)

# 计算准确率
accuracy = calculate_accuracy(results)
print(f"Accuracy: {accuracy * 100:.2f}%")

model_name = model_path.split("/")[-1]
# 保存结果
with open(f'./PubMedQA/results/{model_name}.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
    print("测试完成...")