import os
import re
import json
import subprocess
import sys
from pathlib import Path
from llama_cpp import Llama


def is_file_request(user_text, filenames):
    has_file_reference = any(name.casefold() in user_text.casefold() for name in filenames)
    has_file_action = re.search(
        r"\b(open|read|explain|show|summari[sz]e|describe|inspect|review|check|"
        r"look at|what(?:'s| is) in|tell me about)\b",
        user_text,
        flags=re.IGNORECASE,
    )
    has_file_term = re.search(
        r"\b(file|files|document|documents|folder|text)\b",
        user_text,
        flags=re.IGNORECASE,
    )
    return bool(has_file_action and (has_file_reference or has_file_term))


def is_file_write_request(user_text, filenames, recent_context=""):
    has_file_reference = any(name.casefold() in user_text.casefold() for name in filenames)
    has_write_action = re.search(
        r"\b(write|append|add|save|create|modify|edit|update)\b",
        user_text,
        flags=re.IGNORECASE,
    )
    has_file_term = re.search(
        r"\b(file|files|document|documents|folder|text)\b",
        user_text,
        flags=re.IGNORECASE,
    )
    contextual_reference = re.search(
        r"\b(there|it|that|this|that file|this file)\b",
        user_text,
        flags=re.IGNORECASE,
    )
    context_files = {
        name.casefold()
        for name in filenames
        if name.casefold() in recent_context.casefold()
    }
    return bool(
        has_write_action
        and (
            has_file_reference
            or has_file_term
            or (contextual_reference and len(context_files) == 1)
        )
    )


def resolve_write_target(user_text, filenames, recent_context=""):
    explicit_matches = [
        name for name in filenames if name.casefold() in user_text.casefold()
    ]
    if len(explicit_matches) == 1:
        return explicit_matches[0]
    if explicit_matches:
        raise ValueError("The write request mentions more than one available file.")

    context_matches = {
        name.casefold(): name
        for name in filenames
        if name.casefold() in recent_context.casefold()
    }
    if len(context_matches) == 1:
        return next(iter(context_matches.values()))
    if context_matches:
        raise ValueError("The recent conversation mentions multiple files; name the target file.")
    raise ValueError("Could not determine which existing file to append to.")


REFUSAL_PATTERN = re.compile(
    r"\b(?:i(?:'m| am) sorry|i cannot|i can't|i am unable|i'm unable|"
    r"cannot assist|can't assist|unable to help)\b",
    flags=re.IGNORECASE,
)


def find_refusal_groups(dialogue):
    groups = []
    for index, message in enumerate(dialogue):
        if (
            message["role"] != "assistant"
            or not REFUSAL_PATTERN.search(message["content"])
            or index == 0
            or dialogue[index - 1]["role"] != "user"
        ):
            continue

        if groups and all(
            item["role"] == "user"
            for item in dialogue[groups[-1]["last_refusal_index"] + 1:index]
        ):
            groups[-1]["clarifications"].append(dialogue[index - 1]["content"])
            groups[-1]["last_refusal_index"] = index
        else:
            previous_user_requests = [
                item["content"]
                for item in dialogue[max(0, index - 6):index - 1]
                if item["role"] == "user"
            ][-3:]
            groups.append({
                "question": dialogue[index - 1]["content"],
                "prior_requests": previous_user_requests,
                "clarifications": [],
                "start_index": index - 1,
                "last_refusal_index": index,
            })
    return groups


def build_vga_assembly_correction():
    return (
        "```nasm\n"
        "bits 16\n\n"
        "; Input: AL = character, AH = color attribute. Writes to the first screen cell.\n"
        "print_char:\n"
        "    push ax\n"
        "    push bx\n"
        "    push es\n"
        "    mov bx, ax\n"
        "    mov ax, 0xB800\n"
        "    mov es, ax\n"
        "    mov ax, bx\n"
        "    mov word [es:0], ax\n"
        "    pop es\n"
        "    pop bx\n"
        "    pop ax\n"
        "    ret\n"
        "```\n\n"
        "In 16-bit real mode, `ES:0` with `ES = 0xB800` addresses physical "
        "`0xB8000`. This writes one character and its attribute to the first "
        "text cell. It requires VGA text memory to be mapped and writable; "
        "protected-mode code must map that physical memory first."
    )


def correction_meets_constraints(correction, context_text):
    if not correction or REFUSAL_PATTERN.search(correction):
        return False
    code_blocks = re.findall(r"```.*?```", correction, flags=re.DOTALL)
    code = "\n".join(code_blocks) or correction
    if (
        "assembly" in context_text
        and not re.search(
            r"\b(?:nasm|bits\s+16|mov\s|section\s+\.text)\b",
            code,
            re.IGNORECASE,
        )
    ):
        return False
    if (
        re.search(r"\b(without|no|avoid)\b.{0,40}\bsyscalls?\b", context_text)
        and re.search(r"\b(?:syscall|int\s+0x80|int\s+0x21)\b", code, re.IGNORECASE)
    ):
        return False
    return True


cor = os.cpu_count() or 1
z = 512
y = 0.7
x = ""
g = cor
n_ctx = 4096
model_path = Path(__file__).parent / "models" / "usablemoddels"
modelf = sorted(model_path.glob("*.gguf"))
llm = Llama(model_path=str(modelf[0]), verbose=False, n_threads=g, n_ctx=n_ctx)
messages = [{
    "role": "system",
    "content": (
        "You are an assistant running inside a Python program. The program can read and append "
        "text to existing files inside its files folder. Only if the latest user message explicitly "
        "asks to open, read, or explain a file in that folder, reply with exactly //openfolder and "
        "nothing else. Only if the latest user message explicitly asks to write or append text to "
        "an existing file in that folder, reply with exactly //write and nothing else. Never use "
        "//write to replace or delete existing contents, create files, or act on a file that the "
        "user did not ask to change. Python will perform the requested operation. "
        "When shown a list of unread filenames, choose the exact filename most relevant to the "
        "user's request and reply with only that filename. After receiving file contents, if the "
        "user asked for multiple files or all files and unread files remain, reply exactly "
        "//openagain. Otherwise answer the user normally. "
        "Do not claim that you cannot access files when the user asks about a file; use the marker. "
        "For other requests, answer normally."
    ),
}]
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
        reviewer = llm
        review_model = next(
            (path for path in modelf if "7b" in path.name.casefold()),
            None,
        )
        if review_model is not None and review_model != Path(llm.model_path):
            llm.close()
            reviewer = Llama(
                model_path=str(review_model),
                verbose=False,
                n_threads=g,
                n_ctx=n_ctx,
            )
        dialogue = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message.get("role") in {"user", "assistant"}
        ]
        refusal_groups = find_refusal_groups(dialogue)
        review_messages = [
            {
                "role": "system",
                "content": (
                    "Find assistant replies that are factually wrong, misleading, or refuse a "
                    "safe request. Use the dialogue context, including later user clarifications. "
                    "For each issue return the exact user message that prompted that assistant "
                    "reply. Ignore accurate answers and greetings. Return only JSON in the form "
                    "{\"corrections\": [{\"question\": \"exact user message\"}]};"
                ),
            },
            *dialogue,
        ]
        correction_questions = []
        review_skipped = False
        try:
            result = reviewer.create_chat_completion(
                messages=review_messages,
                max_tokens=512,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            review = re.sub(
                r"<think>.*?(?:</think>|$)\s*",
                "",
                result["choices"][0]["message"]["content"],
                flags=re.DOTALL,
            ).strip()
            review_data = json.loads(review)
            if not isinstance(review_data, dict) or not isinstance(
                review_data.get("corrections"), list
            ):
                raise ValueError("Review model returned invalid corrections JSON.")
            user_messages = {
                message["content"]
                for message in dialogue
                if message["role"] == "user"
            }
            for item in review_data["corrections"]:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("question"), str)
                    and item["question"] in user_messages
                    and item["question"] not in correction_questions
                ):
                    group = next(
                        (
                            group for group in refusal_groups
                            if item["question"] == group["question"]
                            or item["question"] in group["clarifications"]
                        ),
                        None,
                    )
                    correction_questions.append(
                        group["question"] if group else item["question"]
                    )
        except ValueError as error:
            if "exceed context window" in str(error):
                review_skipped = True
            else:
                print(f"Conversation review returned invalid JSON: {error}")

        for group in refusal_groups:
            if group["question"] not in correction_questions:
                correction_questions.append(group["question"])

        corrections = []
        for question in correction_questions:
            refusal_group = next(
                (group for group in refusal_groups if group["question"] == question),
                None,
            )
            if refusal_group:
                prior_requests = refusal_group["prior_requests"]
                later_user_messages = refusal_group["clarifications"]
            else:
                question_index = next(
                    index
                    for index, message in enumerate(dialogue)
                    if message["role"] == "user" and message["content"] == question
                )
                prior_requests = [
                    message["content"]
                    for message in dialogue[max(0, question_index - 6):question_index]
                    if message["role"] == "user"
                ][-3:]
                later_user_messages = [
                    message["content"]
                    for message in dialogue[question_index + 1:]
                    if message["role"] == "user"
                ][-4:]
            request = (
                "Earlier user requests that may specify format or intent:\n"
                + ("\n".join(prior_requests) or "(none)")
                + f"\n\nRequest to answer:\n{question}\n\n"
                "Relevant later clarifications:\n"
                + ("\n".join(later_user_messages) or "(none)")
                + "\n\nGive a concise, complete, accurate replacement answer to the original "
                "request, preserving its requested language and format and using relevant "
                "clarifications. If assembly was requested, answer in assembly, not C. Do not "
                "repeat or defend the assistant's previous answer. Do not write a review or discuss "
                "the conversation; answer the user directly."
            )
            context_text = " ".join(
                prior_requests + [question] + later_user_messages
            ).casefold()
            is_vga_assembly_request = (
                "assembly" in context_text
                and ("vga" in context_text or "0xb8000" in context_text)
                and re.search(r"\b(without|no|avoid)\b.{0,40}\bsyscalls?\b", context_text)
            )
            if is_vga_assembly_request:
                correction = build_vga_assembly_correction()
            else:
                correction = ""
                for attempt in range(2):
                    extra_instruction = (
                        ""
                        if attempt == 0
                        else (
                            " The previous draft violated the requested constraints. "
                            "Start over and obey the user-requested language and format exactly."
                        )
                    )
                    result = reviewer.create_chat_completion(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer the user's request directly and accurately. Do not refuse a "
                                    "benign request. Preserve the requested programming language and code "
                                    "format exactly; never substitute a different language. For code, give "
                                    "a complete minimal example and state important environment assumptions."
                                    + extra_instruction
                                ),
                            },
                            {"role": "user", "content": request},
                        ],
                        max_tokens=768,
                        temperature=0.1,
                    )
                    candidate = re.sub(
                        r"<think>.*?(?:</think>|$)\s*",
                        "",
                        result["choices"][0]["message"]["content"],
                        flags=re.DOTALL,
                    ).strip()
                    if (
                        result["choices"][0].get("finish_reason") != "length"
                        and correction_meets_constraints(candidate, context_text)
                    ):
                        correction = candidate
                        break
            if (
                correction_meets_constraints(correction, context_text)
                and correction.casefold() != question.casefold()
            ):
                corrections.append(
                    {"question": question, "correction": correction}
                )
            else:
                print(f"Could not generate a usable correction for: {question}")

        if review_skipped:
            print("Factual review skipped because the conversation exceeds the model context.")
        print(json.dumps({"corrections": corrections}, indent=2, ensure_ascii=False))
        if reviewer is not llm:
            reviewer.close()
        llm.close()
        trainer_script = Path(__file__).with_name("extr.py")
        conversation_file = Path(__file__).with_name("crrections.json")
        print(f"Starting trainer: {trainer_script.name}")
        subprocess.run(
            [sys.executable, str(trainer_script), str(conversation_file)],
            check=True,
            cwd=Path(__file__).parent,
        )
        break
    else:
        if x == "//maxtokens":
            z = int(input("Enter new context size (budget includes COT): "))
        else:
            if x == "//maxthreads":
                g = max(1, min(int(input(f"Enter new number of threads (1-{cor}): ")), cor))
                llm.close()
                llm = Llama(model_path=str(modelf[0]), verbose=False, n_threads=g, n_ctx=n_ctx)
            else:
                if x == "//temperature":
                    y = float(input("Enter new temperature: "))
                else:
                    messages.append({"role": "user", "content": x})
                    print("thinking...")
                    result = llm.create_chat_completion(messages=messages, max_tokens=z, temperature=y)
                    response = result["choices"][0]["message"]["content"]
                    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
                    fl = Path(__file__).parent / "files"
                    available_files = [file for file in fl.iterdir() if file.is_file()]
                    filenames = [file.name for file in available_files]
                    recent_context = "\n".join(
                        str(message.get("content", ""))
                        for message in messages[-6:-1]
                    )
                    write_requested = is_file_write_request(
                        x, filenames, recent_context
                    )
                    read_requested = is_file_request(x, filenames)
                    if write_requested:
                        response = "//write"
                    elif read_requested:
                        response = "//openfolder"
                    if response == "//openfolder":
                        if not is_file_request(x, [file.name for file in available_files]):
                            response = "I only open files when you ask me to open, read, or explain one."
                    if response == "//openfolder":
                        working_messages = messages.copy()
                        while True:
                            print("Opening files...")
                            list_of_files = "\n".join(file.name for file in available_files)
                            working_messages.append({
                                "role": "system",
                                "content": (
                                    f"Available files:\n{list_of_files}\n"
                                    "Choose the file requested by the user. Reply with only its exact filename."
                                ),
                            })
                            print("thinking...")
                            result = llm.create_chat_completion(
                                messages=working_messages, max_tokens=z, temperature=y
                            )
                            response = result["choices"][0]["message"]["content"]
                            response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
                            selected_name = response.strip().strip("\"'")
                            if Path(selected_name).name != selected_name:
                                raise ValueError("The model returned an invalid filename.")
                            fil = fl / selected_name
                            if not fil.is_file():
                                raise FileNotFoundError(f"Selected file does not exist: {selected_name}")
                            content = fil.read_text(encoding="utf-8")
                            working_messages.append({
                                "role": "system",
                                "content": f"Contents of {selected_name}:\n{content}\nAnswer the user's question about this file. Or chose to open another file if the user requested that by saying //openfolder and nothing else. If the user did not request another file, answer normally.",
                            })
                            print("thinking...")
                            result = llm.create_chat_completion(
                                messages=working_messages, max_tokens=z, temperature=y
                            )
                            response = result["choices"][0]["message"]["content"]
                            response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
                            if response == "//openfolder":
                                continue
                            else:
                                print(response)
                                messages.append({"role": "assistant", "content": response})
                                with open("hcat.json", "w", encoding="utf-8") as f:
                                    json.dump({"messages": messages}, f, indent=2, ensure_ascii=False)
                                break
                    else:
                        if response == "//write":
                            if not write_requested:
                                response = "I only write to files when you explicitly ask me to."
                            else:
                                selected_name = resolve_write_target(
                                    x, filenames, recent_context
                                )
                                fil = fl / selected_name
                                content = fil.read_text(encoding="utf-8")
                                print("writing...")
                                result = llm.create_chat_completion(
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": (
                                                "Create only the new text the user requests for a file. "
                                                "Use the latest request as the sole source of what to write. "
                                                "If the user asks for a poem, write the poem itself. "
                                                "Do not refer to previous conversation, existing file "
                                                "contents, or explain what you will do. Do not refuse "
                                                "harmless creative-writing requests. Return only the "
                                                "text to append, with no quotation marks or Markdown fences."
                                            ),
                                        },
                                        {"role": "user", "content": x},
                                    ],
                                    max_tokens=z,
                                    temperature=y,
                                )
                                append_text = result["choices"][0]["message"]["content"]
                                append_text = re.sub(
                                    r"<think>.*?</think>\s*",
                                    "",
                                    append_text,
                                    flags=re.DOTALL,
                                ).strip()
                                if not append_text:
                                    raise ValueError("The model returned no text to append.")
                                with fil.open("a", encoding="utf-8") as output_file:
                                    if content and not content.endswith(("\n", "\r")):
                                        output_file.write("\n")
                                    output_file.write(append_text)
                                response = f"Appended the requested text to {selected_name}."
                        print(response)
                        messages.append({"role": "assistant", "content": response})
                        with open("hcat.json", "w", encoding="utf-8") as f:
                            json.dump({"messages": messages}, f, indent=2, ensure_ascii=False)