import os
from pathlib import Path
from llama_cpp import Llama
model_dir = Path(__file__).parent / "models" / "usablemoddels"
modelf = sorted(model_dir.glob("*.gguf"))
if not modelf:
    print("No usable models found in the 'models/usablemoddels' directory.")
    exit(1)
else:
    print(modelf)
print("sofai startin~")
while True:
    x = input("usah>>> ")
    compare = x == "//exit"
    if true == compare:
        print("bye bye~")
        break
    else:
        print(x)