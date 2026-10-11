import gc
import shutil
import os
import re
import subprocess
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
from llama_cpp import Llama
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime_paths import DATA_DIR
def collate(examples, tokenizer):
    batch = tokenizer.pad(
        [{"input_ids": example["input_ids"]} for example in examples],
        padding=True,
        return_tensors="pt",
    )
    max_length = batch["input_ids"].shape[1]
    batch["labels"] = torch.tensor(
        [
            example["labels"] + [-100] * (max_length - len(example["labels"]))
            for example in examples
        ],
        dtype=torch.long,
    )
    return batch


def load_training_examples(corrections, tokenizer):
    if not tokenizer.chat_template:
        raise ValueError("The selected tokenizer does not define a chat template.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("The selected tokenizer has no padding token or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized_examples = []
    for example in corrections:
        user_messages = [{"role": "user", "content": example["question"]}]
        full_messages = user_messages + [
            {"role": "assistant", "content": example["correction"]}
        ]
        prompt_ids = tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "This tokenizer's chat template does not preserve the prompt as a "
                "prefix of the complete conversation; cannot safely mask prompt labels."
            )
        tokenized_examples.append(
            {
                "input_ids": full_ids,
                "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
            }
        )
    return tokenized_examples
def train_model(project_dir, gguf_name, corrections):
    gguf_file = Path(gguf_name)
    if gguf_file.name != gguf_name or gguf_file.suffix.casefold() != ".gguf":
        raise ValueError(f"Invalid GGUF filename in correction data: {gguf_name!r}")

    gguf_path = project_dir / "models" / "usablemoddels" / gguf_file.name
    if not gguf_path.is_file():
        raise FileNotFoundError(f"The chat model does not exist: {gguf_path}")

    training_model_path = (
        project_dir / "models" / "trainable_models" / gguf_file.stem
    )
    if not training_model_path.is_dir():
        raise FileNotFoundError(
            f"Put the matching Hugging Face checkpoint for {gguf_file.name} "
            f"in {training_model_path}."
        )
    print(f"Preparing to train {gguf_file.name}", flush=True)
    print(f"Checkpoint: {training_model_path}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        training_model_path,
        local_files_only=True,
    )
    print(f"Tokenizer loaded; processing {len(corrections)} correction(s)...", flush=True)
    tokenized_examples = load_training_examples(corrections, tokenizer)
    loader = DataLoader(
        tokenized_examples,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda examples: collate(examples, tokenizer),
    )
    print(
        f"Prepared {len(tokenized_examples)} example(s) "
        f"({len(loader)} batch(es) per epoch).",
        flush=True,
    )

    print("Loading base model weights (this may take a while)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        training_model_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    print("Base model weights loaded; attaching LoRA adapters...", flush=True)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
        ),
    )
    model.to("cpu")
    model.train()
    print("LoRA adapters attached; model is ready on CPU.", flush=True)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=2e-4,
    )
    epochs = len(corrections) * 2
    total_steps = epochs * len(loader)
    completed_steps = 0
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        print(f"Starting epoch {epoch + 1}/{epochs}...", flush=True)
        for batch_index, batch in enumerate(loader, start=1):
            step_start = time.perf_counter()
            input_tokens = batch["input_ids"].numel()
            print(
                f"Epoch {epoch + 1}/{epochs}, batch {batch_index}/{len(loader)} "
                f"({completed_steps}/{total_steps} steps complete): "
                f"running CPU forward pass on {input_tokens} token(s)...",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            if outputs.loss is None:
                raise RuntimeError("The model did not return a training loss.")
            loss = outputs.loss.item()
            print(
                f"Forward pass finished; loss={loss:.4f}. "
                "Running backward pass...",
                flush=True,
            )
            outputs.loss.backward()
            print("Backward pass finished; updating LoRA weights...", flush=True)
            optimizer.step()
            total_loss += loss
            completed_steps += 1
            print(
                f"Step {completed_steps}/{total_steps} complete; "
                f"step took {time.perf_counter() - step_start:.1f}s, "
                f"loss={loss:.4f}.",
                flush=True,
            )
        print(
            f"Epoch {epoch + 1}/{epochs} complete; "
            f"average loss={total_loss / len(loader):.4f}; "
            f"took {time.perf_counter() - epoch_start:.1f}s.",
            flush=True,
        )

    adapter_dir = (
        project_dir
        / "models"
        / "adapters"
        / gguf_file.stem
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    adapter_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving LoRA adapter and tokenizer to {adapter_dir}...", flush=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}", flush=True)
    merged_model = model.merge_and_unload(safe_merge=True)
    mergedmodeldir = (
        project_dir
        / "models"
        / "mergedmodels"
        / gguf_file.stem
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    mergedmodeldir.mkdir(parents=True, exist_ok=False)
    print(f"Saving merged model to {mergedmodeldir}...", flush=True)
    merged_model.save_pretrained(mergedmodeldir, safe_serialization=True)
    tokenizer.save_pretrained(mergedmodeldir)

    llama_cpp_dir = Path(os.environ["LLAMA_CPP_DIR"])
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantizer = Path(os.environ["LLAMA_QUANTIZE_EXE"])
    converter_python = Path(
        os.environ.get("SOFAI_CONVERTER_PYTHON", sys.executable)
    )
    if not converter.is_file():
        raise FileNotFoundError(f"llama.cpp converter not found: {converter}")
    if not quantizer.is_file():
        raise FileNotFoundError(f"llama.cpp quantizer not found: {quantizer}")
    if not converter_python.is_file():
        raise FileNotFoundError(
            f"Python interpreter for GGUF conversion not found: {converter_python}"
        )
    if getattr(sys, "frozen", False) and not os.environ.get(
        "SOFAI_CONVERTER_PYTHON"
    ):
        raise RuntimeError(
            "Set SOFAI_CONVERTER_PYTHON to a Python interpreter with the "
            "llama.cpp conversion dependencies before using the packaged trainer."
        )

    converted_dir = (
        project_dir
        / "models"
        / "convertedmodels"
        / gguf_file.stem
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    converted_dir.mkdir(parents=True, exist_ok=False)
    f16_gguf = converted_dir / f"{gguf_file.stem}-f16.gguf"
    quantized_gguf = converted_dir / gguf_file.name
    quantization_match = re.search(
        r"(Q\d+_K_[SML]|Q\d+_\d+)$",
        gguf_file.stem,
        flags=re.IGNORECASE,
    )
    if not quantization_match:
        raise ValueError(
            f"Cannot determine quantization type from model filename: {gguf_file.name}"
        )
    quantization_type = quantization_match.group(1).upper()

    subprocess.run(
        [
            str(converter_python),
            str(converter),
            str(mergedmodeldir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(quantizer),
            str(f16_gguf),
            str(quantized_gguf),
            quantization_type,
        ],
        check=True,
    )
    print(f"Quantized GGUF saved to: {quantized_gguf}", flush=True)
    print(f"Testing converted GGUF: {quantized_gguf}", flush=True)
    testmodel = Llama(
        model_path=str(quantized_gguf),
        n_ctx=512,
        n_threads=1,
        verbose=False,
    )
    testmodel.close()
    backup_dir = project_dir / "models" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / (
        f"{gguf_file.name}.{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.bak"
    )
    shutil.copy2(gguf_path, backup_path)
    os.replace(quantized_gguf, gguf_path)
    print(f"Original GGUF backed up to: {backup_path}", flush=True)
    print(f"Replaced model at: {gguf_path}", flush=True)
    del optimizer, model, merged_model, tokenizer, loader, tokenized_examples
    gc.collect()


def main():
    project_dir = DATA_DIR
    if len(sys.argv) > 2:
        raise SystemExit("Usage: extr.py [corrections.json]")
    corrections_file = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) == 2
        else project_dir / "corrections.json"
    )
    if not corrections_file.is_file():
        raise SystemExit(
            f"Corrections file not found: {corrections_file}\n"
            "Run core.py and exit with //exit to generate corrections.json, "
            "or pass an existing corrections JSON path to extr.py."
        )
    corrections_text = corrections_file.read_text(encoding="utf-8").strip()
    correction_data = json.loads(corrections_text) if corrections_text else []
    corrections = (
        correction_data.get("corrections", [])
        if isinstance(correction_data, dict)
        else correction_data
    )
    if not isinstance(corrections, list):
        raise ValueError(f"Expected a list of corrections in {corrections_file}.")

    legacy_model_file = project_dir / "whichmd.txt"
    legacy_model_name = (
        legacy_model_file.read_text(encoding="utf-8").strip()
        if legacy_model_file.is_file()
        else ""
    )
    corrections_by_model = {}
    used_legacy_model = False
    for item in corrections:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("question"), str)
            or not item["question"].strip()
            or not isinstance(item.get("correction"), str)
            or not item["correction"].strip()
        ):
            continue
        model_name = item.get("model")
        if model_name is None:
            if not legacy_model_name:
                raise ValueError(
                    "A correction has no model field, and whichmd.txt is missing or empty. "
                    "Run core.py to select a model, then regenerate corrections."
                )
            model_name = legacy_model_name
            used_legacy_model = True
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "The correction's model field must be a non-empty GGUF filename."
            )
        corrections_by_model.setdefault(model_name, []).append(item)

    if used_legacy_model:
        print(
            "Warning: legacy correction(s) have no model attribution; "
            f"using the currently selected model from whichmd.txt ({legacy_model_name})."
        )

    if not corrections_by_model:
        raise ValueError(f"No valid question/correction examples found in {corrections_file}.")

    for model_name, model_corrections in corrections_by_model.items():
        train_model(project_dir, model_name, model_corrections)


if __name__ == "__main__":
    main()
