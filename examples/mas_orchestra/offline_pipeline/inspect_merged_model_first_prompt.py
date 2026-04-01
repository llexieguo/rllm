from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)

from examples.mas_orchestra.offline_replay import load_offline_replay_file


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif item_type == "input_text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return ""
    return str(content)


def _strip_messages_to_text(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    stripped: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = _extract_text(message.get("content"))
        stripped.append({"role": role, "content": text})
    return stripped


def _format_messages(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if not content:
            continue
        chunks.append(f"[{role}]\n{content}")
    return "\n\n".join(chunks)


def _load_model(model_path: str, trust_remote_code: bool, dtype: torch.dtype):
    load_errors: list[str] = []
    for model_cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            return model_cls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:  # pragma: no cover - best effort fallback chain
            load_errors.append(f"{model_cls.__name__}: {exc}")
    raise RuntimeError("Failed to load model:\n" + "\n".join(load_errors))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a merged HF model, read the first offline replay prompt, and print one response."
    )
    parser.add_argument("--model-path", required=True, help="Merged Hugging Face model directory.")
    parser.add_argument("--train-file", required=True, help="Offline replay train parquet/jsonl/json directory.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = load_offline_replay_file(args.train_file)
    if not rows:
        raise RuntimeError(f"No rows found in training data: {args.train_file}")

    sample = rows[0]
    step = next((item for item in sample["steps"] if item.get("trainable")), sample["steps"][0])
    messages = _strip_messages_to_text(step["messages"])

    prompt_preview = _format_messages(messages)
    print("=== First Prompt (text only) ===")
    print(prompt_preview)
    print()

    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    model = _load_model(args.model_path, trust_remote_code=args.trust_remote_code, dtype=dtype)
    model.to(args.device)
    model.eval()

    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor(text=[prompt], return_tensors="pt")
    except Exception:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(prompt, return_tensors="pt")

    model_inputs = {key: value.to(args.device) for key, value in model_inputs.items()}

    with torch.no_grad():
        generated = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)

    prompt_len = model_inputs["input_ids"].shape[1]
    output_ids = generated[0, prompt_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    print("=== Model Output ===")
    print(output_text)


if __name__ == "__main__":
    main()
