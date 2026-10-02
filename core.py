import os
import re
import json
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


cor = os.cpu_count() or 1
z = 512
y = 0.7
x = ""
g = cor
model_path = Path(__file__).parent / "models" / "usablemoddels"
modelf = sorted(model_path.glob("*.gguf"))
llm = Llama(model_path=str(modelf[0]), verbose=False, n_threads=g)
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
            if x == "//maxthreads":
                g = max(1, min(int(input(f"Enter new number of threads (1-{cor}): ")), cor))
                llm.close()
                llm = Llama(model_path=str(modelf[0]), verbose=False, n_threads=g)
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