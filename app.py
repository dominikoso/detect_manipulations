import csv
import ollama
import os
from datetime import datetime

def load_data(csv_path: str):
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "text": row["Text"],     
                "label": row["Label"]    
            })
    return data

def analyze_sentence(sentence: str, model_name: str = "deepseek-r1:14b") -> str:
    client = ollama.Client()
    prompt = f"""
You are a text classification tool tasked with determining if a sentence uses manipulative language. Manipulative language includes tactics like emotional appeals, guilt-tripping, undue flattery, or any covert persuasion aimed at influencing behavior or opinions.

Classify the given sentence using ONLY ONE WORD:
- "MANIPULATIVE" if the sentence employs such tactics.
- "NOT_MANIPULATIVE" if it does not.

Sentence: "{sentence}"
"""
    result = client.generate(model=model_name, prompt=prompt)
    d=datetime.now()
    os.makedirs(f"results/test-{d.strftime('%Y_%m_%d')}", exist_ok=True)
    open(
        os.path.join(f"results/test-{d.strftime('%Y_%m_%d')}", 
                     f"{model_name}_{d.strftime('%Y_%m_%d-%I_%M_%S_%p')}_result.txt"), "w").write(prompt+"\n"+result["response"]
        )
    return result['response'].splitlines()[-1]

def main():
    dataset = load_data("test_data.csv")
    correct = 0
    isCorrect = False

    for i, item in enumerate(dataset, start=1):
        text = item["text"]
        true_label = item["label"]
        print(f"[*] Analyzing text \"{text}\" with true label: {true_label}")
        model_output = analyze_sentence(text, model_name="deepseek-r1:14b")
        if model_output == true_label:
            isCorrect = True
            correct += 1
        else:
            isCorrect = False

        print("[*] Result of analysis:")
        print(f"    ID: {i}")
        print(f"    Text: {text}")
        print(f"    True Label: {true_label}")
        print(f"    Model Predition: {model_output}")
        print(f"    Was model correct: {isCorrect}")
        print("-" * 60)

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
