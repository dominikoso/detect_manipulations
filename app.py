import csv
import os
from datetime import datetime
from typing import Tuple

import ollama


def load_data(csv_path: str):
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"text": row["Text"], "label": row["Label"]})
    return data


def analyze_sentence(
    sentence: str, filedir: str, model_name: str = "deepseek-r1:14b"
) -> Tuple[str, str]:
    client = ollama.Client()
    prompt = f"""
You are a text classification tool tasked with determining if a sentence uses manipulative language. Manipulative language includes tactics like emotional appeals, guilt-tripping, undue flattery, or any covert persuasion aimed at influencing behavior or opinions.

Classify the given sentence using ONLY ONE WORD, do not add any special characters:
- "MANIPULATIVE" if the sentence employs such tactics.
- "NOT_MANIPULATIVE" if it does not.

Sentence: "{sentence}"
"""
    d = datetime.now()
    result = client.generate(model=model_name, prompt=prompt)
    filename = os.path.join(
        filedir, f"{model_name}_{d.strftime('%Y_%m_%d-%I_%M_%S_%p')}_result.txt"
    )
    open(filename, "w").write(prompt + "\n" + result["response"])
    return (result["response"].splitlines()[-1], filename)


def perform_test_per_model(dataset: list, model_name: str, filedir: str):
    correct = 0
    isCorrect = False

    for i, item in enumerate(dataset, start=1):
        text = item["text"]
        true_label = item["label"]
        print(
            f'[*] [{model_name}] Analyzing text "{text}" with true label: {true_label}'
        )
        model_output, filename = analyze_sentence(text, filedir, model_name)
        if model_output.replace(" ", "").replace("NOT_", "NOT_MANIPULATIVE").upper() == true_label.upper():
            isCorrect = True
            correct += 1
        else:
            isCorrect = False
        with open(filename, "+a") as f:
            f.write("\n\n[*] Result of analysis:\n")
            f.write(f"    ID: {i}\n")
            f.write(f"    Text: {text}\n")
            f.write(f"    True Label: {true_label}\n")
            f.write(f"    Model Predition: {model_output}\n")
            f.write(f"    Was model correct: {isCorrect}\n")
            f.write("-" * 60 + "\n")

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0
    print(f"[*] Accuracy for model {model_name}: {accuracy:.2f}")


def main():
    dataset = load_data("test_data.csv")
    models = [
        "deepseek-r1:14b",
        "deepseek-r1:32b",
    ]
    d = datetime.now()
    filedir = f"results/test-{d.strftime('%Y_%m_%d_%I_%M')}"
    os.makedirs(filedir, exist_ok=True)
    for model_name in models:
        perform_test_per_model(dataset, model_name, filedir)


if __name__ == "__main__":
    main()
