import csv
import ollama
import os
from typing import Tuple
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

def analyze_sentence(sentence: str, model_name: str = "deepseek-r1:14b") -> Tuple[str, str]:
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
    os.makedirs(f"results/test-{d.strftime('%Y_%m_%d_%I_%M')}", exist_ok=True)
    filename = os.path.join(f"results/test-{d.strftime('%Y_%m_%d_%I_%M')}", 
                            f"{model_name}_{d.strftime('%Y_%m_%d-%I_%M_%S_%p')}_result.txt")
    open(filename, "w").write(prompt+"\n"+result["response"])
    return (result['response'].splitlines()[-1], filename)

def main():
    dataset = load_data("test_data.csv")
    model_name="deepseek-r1:14b"
    correct = 0
    isCorrect = False

    for i, item in enumerate(dataset, start=1):
        text = item["text"]
        true_label = item["label"]
        print(f"[*] Analyzing text \"{text}\" with true label: {true_label}")
        model_output, filename = analyze_sentence(text, model_name)
        if model_output == true_label:
            isCorrect = True
            correct += 1
        else:
            isCorrect = False
        with open(filename, "+a") as f:
            f.write("\n[*] Result of analysis:\n")
            f.write(f"    ID: {i}\n")
            f.write(f"    Text: {text}\n")
            f.write(f"    True Label: {true_label}\n")
            f.write(f"    Model Predition: {model_output}\n")
            f.write(f"    Was model correct: {isCorrect}\n")
            f.write("-" * 60 + "\n")

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0
    print(f"[*] Accuracy for model {model_name}: {accuracy:.2f}")

if __name__ == "__main__":
    main()
