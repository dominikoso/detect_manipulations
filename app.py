import csv
import ollama

def load_data(csv_path: str):
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "text": row["Text"],      # e.g. "I think you are incompetent..."
                "label": row["Label"]     # e.g. "MANIPULATIVE" or "NOT_MANIPULATIVE"
            })
    return data

def analyze_sentence(sentence: str, model_name: str = "deepseek-r1:14b") -> str:
    client = ollama.Client()
    prompt = f"""
You are a text classification tool. 
Classify the following sentence as MANIPULATIVE or NOT_MANIPULATIVE, with no additional text:

Sentence: "{sentence}"
"""
    output_text = []
    for event in client.generate(model=model_name, prompt=prompt):
        if event["type"] == "completion":
            output_text.append(event["data"])
    final_text = "".join(output_text).strip()
    return final_text

def main():
    dataset = load_data("test_data.csv")
    correct = 0

    for i, item in enumerate(dataset, start=1):
        text = item["text"]
        true_label = item["label"]
        model_output = analyze_sentence(text, model_name="llama2")

        if "MANIP" in model_output.upper():
            predicted_label = "MANIPULATIVE"
        else:
            predicted_label = "NOT_MANIPULATIVE"

        if predicted_label == true_label:
            correct += 1

        print(f"ID: {i}")
        print(f"Text: {text}")
        print(f"True Label: {true_label}")
        print(f"Model Output: {model_output}")
        print(f"Predicted: {predicted_label}")
        print("-" * 60)

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
