import os
import re
import json
from pathlib import Path
from llama_cpp import Llama
z = 512
y = 0.7
x = ""
model_path = Path(__file__).parent / "models" / "usablemoddels"
modelf = sorted(model_path.glob("*.gguf"))
llm = Llama(model_path=str(modelf[0]), verbose=False)
messages = []
if not modelf:
    print("No usable models found in the 'models/usablemoddels' directory.")
    exit(1)
else:
    print(modelf)
print("sofai startin~")
while True:
    x = input("usah>>> ")
    if x == "//exit":
        print("bye bye~")
        with open("hcat.json", "w", encoding="utf-8") as h:
            json.dump({"messages": messages}, h, indent=2, ensure_ascii=False)
        review_messages = messages + [{
            "role": "user",
            "content": (
                "Review the conversation above. Find only assistant messages that are incorrect "
                "or misleading, and correct them. Return a JSON array only, with no Markdown or "
                "explanation. Each item must have exactly these keys: \"question\" and \"correction\". "
                "Put the original user question in \"question\" and the corrected assistant answer "
                "in \"correction\". Return [] if every assistant answer is correct."
            )
        }]
        result = llm.create_chat_completion(messages=review_messages, max_tokens=4096, temperature=y)
        review = result["choices"][0]["message"]["content"]
        review = re.sub(r"<think>.*?(?:</think>|$)\s*", "", review, flags=re.DOTALL).strip()
        print(review)
        break
    else:
        if x == "//maxtokens":
            z = int(input("Enter new context size (budget includes COT): "))
        else:
            if x == "//temperature":
                y = float(input("Enter new temperature: "))
            else:
                messages.append({"role": "user", "content": x})
                result = llm.create_chat_completion(messages=messages, max_tokens=z, temperature=y)
                response = result["choices"][0]["message"]["content"]
                response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
                print(response)
                messages.append({"role": "assistant", "content": response})
                with open("hcat.json", "w", encoding="utf-8") as f:
                    json.dump({"messages": messages}, f, indent=2, ensure_ascii=False)
