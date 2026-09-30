import os
import re
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
    compare = x == "//exit"
    if True == compare:
        print("bye bye~")
        break
    else:
        compare = x == "//maxtokens"
        if compare:
            z = int(input("Enter new context size (budget includes COT): "))
        else:
            compare = x == "//temperature"
            if compare:
                y = float(input("Enter new temperature: "))
            else:
                messages.append({"role": "user", "content": x})
                result = llm.create_chat_completion(messages=messages, max_tokens=z, temperature=y)
                response = result["choices"][0]["message"]["content"]
                response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
                print(response)
                messages.append({"role": "assistant", "content": response})