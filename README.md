# sofai
client for chating with a local GGUF model. When corrections are generated at exit, sofai trains
a LoRA adapter for the model that produced each original wrong answer. then it merges the modell and makes a new one which knows the fixed answer.

Install PyTorch's CPU build + the runtime/training libraries in the
virtual environment:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers peft llama-cpp-python
```

## Adding models for chat and training

For each chat model, provide both:

- Its GGUF file in `models/usablemoddels/`.
- The matching original Hugging Face checkpoint in
  `models/trainable_models/<GGUF filename without .gguf>/`.

For example, `models/usablemoddels/example.gguf` pairs with the checkpoint
directory `models/trainable_models/example/`. The checkpoint directory must
contain the model weights, configuration, tokenizer files, and a chat template.
The GGUF is used only for chatting; it is not trained directly.

Use `//model` to change model
Use `//maxtokens` to change max amount of tokens
Use `//temperature` to change temp
Use `//exit` to exit
