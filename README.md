# detect_manipulations

Manipulation detector based on Local LLM used in Project for Język w Kontekście społecznym.

# Requirements
- Ollama installed (eg. `brew install ollama`) and running (`ollama serve`)
- For full reproduction deepseek-r1:14, mistral and llama3:8b-text installed (`ollama pull <model_name>`)

# Instruction
```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ python3 app.py
```

# Expected output
```
[*] [deepseek-r1:14b] Analyzing text "Man is by nature a political animal." with true label: NOT_MANIPULATIVE
[*] [deepseek-r1:14b] Analyzing text "But if thought corrupts language, language can also corrupt thought." with true label: NOT_MANIPULATIVE
[*] [deepseek-r1:14b] Analyzing text "Politics is the art of the possible, the attainable — the art of the next best." with true label: NOT_MANIPULATIVE
(...)
[*] Accuracy for model deepseek-r1:14b: 0.65
```

# Author
Dominik Kostecki