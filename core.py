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
                "Review the conversation using all its context. Report only assistant replies that "
                "are factually wrong or misleading; greetings and other harmless pleasantries are "
                "not errors. Pair each correction with the exact user message the assistant was "
                "answering, not with a later statement that provides context. If a later user message "
                "provides the answer to an earlier question, use that fact to correct the earlier "
                "answer when appropriate. For example:\n"
                "User: hi\nAssistant: Hi! How can I assist you today?\n"
                "User: what is my name\nAssistant: You have not told me your name.\n"
                "User: my name is superuser\nAssistant: Hello, superuser!\n"
                "The greeting is not an error. The name correction, if needed, belongs to \"what is "
                "my name\" and says \"Your name is Superuser. Basically everywhere that the user corrected you\" Never attach it to \"my name is "
                "superuser\" or flag a greeting as wrong.\n\n"
                "Return a JSON array only, with no Markdown or explanation. Each array item must "
                "have exactly two string keys: \"question\" and \"correction\". \"question\" must "
                "contain the complete original user message that prompted the flawed reply, copied "
                "verbatim. \"correction\" must contain a complete, accurate replacement answer to "
                "that user message. Keep each person's identity and pronouns consistent with the "
                "conversation. Do not include the flawed answer, commentary, or extra keys."
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
