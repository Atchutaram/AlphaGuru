"""
OpenAssistant (OASST1) tokenizer and sharder for SFT training (efficient version).
Each assistant reply is trained with the full multi-turn prompt leading up to it.
"""

import os
import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

# Configuration
OUT_DIR = os.path.join(os.path.dirname(__file__), "oasst_sft")
os.makedirs(OUT_DIR, exist_ok=True)
SHARD_SIZE = 100_000
MAX_LENGTH = 1024

# Custom tokenizer with added special tokens
base = tiktoken.get_encoding("gpt2")
enc = tiktoken.Encoding(
    name="gpt2_custom",
    pat_str=base._pat_str,
    mergeable_ranks=base._mergeable_ranks,
    special_tokens={
        **base._special_tokens,
        "<|user|>": 50257,
        "<|assistant|>": 50258,
    },
)

user_token_seq = enc.encode("<|user|>", allowed_special={"<|user|>", "<|assistant|>"})


def format_message(role, text):
    prefix = "<|user|>" if role == "prompter" else "<|assistant|>"
    return f"{prefix} {text.strip()}"


def extract_pairs_from_tree(messages):
    id_to_msg = {msg["message_id"]: msg for msg in messages}
    children_map = defaultdict(list)
    for msg in messages:
        parent_id = msg.get("parent_id")
        if parent_id:
            children_map[parent_id].append(msg)

    pairs = []

    def dfs(msg, path_texts):
        if msg.get("lang") != "en":
            return  # skip subtrees with non-English content

        text = msg.get("text", "").strip()
        if not text:
            return

        if msg["role"] == "assistant":
            prompt = "\n".join(path_texts)
            response = text
            if response and prompt:
                pairs.append((prompt, response))

        if msg["role"] in {"prompter", "assistant"}:
            formatted = format_message(msg["role"], text)
            new_path = path_texts + [formatted]
        else:
            new_path = path_texts

        for child in children_map.get(msg["message_id"], []):
            dfs(child, new_path)

    # Start DFS from root messages
    for msg in messages:
        if msg["parent_id"] is None:
            dfs(msg, [])

    return pairs


def contains_subsequence(sequence, subseq):
    for i in range(len(sequence) - len(subseq) + 1):
        if sequence[i:i + len(subseq)] == subseq:
            return True
    return False


def tokenize_pair(pair):
    prompt, response = pair
    try:
        prompt_ids = enc.encode(prompt, allowed_special={"<|user|>", "<|assistant|>"})
        response_ids = enc.encode(" " + response, allowed_special={"<|user|>", "<|assistant|>"})
    except Exception:
        return None

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    truncation_happened = False
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[-MAX_LENGTH:]
        labels = labels[-MAX_LENGTH:]
        truncation_happened = True

    if truncation_happened:
        if not contains_subsequence(input_ids, user_token_seq):
            return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def save_shard(shard_data, shard_idx, split):
    filename = os.path.join(OUT_DIR, f"oasst_sft_{split}_{shard_idx:06d}.pt")
    torch.save(shard_data, filename)


def process_and_save(pairs, split):
    print(f"Tokenizing {split} set (serial)...")

    shard = []
    shard_idx = 0

    for pair in tqdm(pairs, desc=f"Processing {split} pairs"):
        result = tokenize_pair(pair)
        if result is not None:
            shard.append(result)
            if len(shard) >= SHARD_SIZE:
                save_shard(shard, shard_idx, split=split)
                shard_idx += 1
                shard = []
    if shard:
        save_shard(shard, shard_idx, split=split)

    print(f"Saved {shard_idx + 1} shard(s) for {split} split.")


def prepare_split(split_name, dataset):
    print(f"Grouping {split_name} messages into trees...")
    grouped = defaultdict(list)
    for msg in dataset:
        grouped[msg["message_tree_id"]].append(msg)

    print(f"Extracting {split_name} prompt-response pairs...")
    pairs = []
    for tree in tqdm(grouped.values(), desc=f"Building {split_name} pairs"):
        pairs.extend(extract_pairs_from_tree(tree))

    print(f"{split_name.capitalize()} size: {len(pairs)}")
    return pairs


def main():
    print("Loading OpenAssistant dataset...")
    train_dataset = load_dataset("OpenAssistant/oasst1", split="train")
    val_dataset = load_dataset("OpenAssistant/oasst1", split="validation")

    train_pairs = prepare_split("train", train_dataset)
    val_pairs = prepare_split("val", val_dataset)

    process_and_save(train_pairs, split="train")
    process_and_save(val_pairs, split="val")

if __name__ == "__main__":
    main()
