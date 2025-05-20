import torch
import torch.nn.functional as F
import numpy as np
import os
import math

from dataclasses import dataclass


@dataclass
class GPTConfig:
    context_len: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    d_model: int = 768

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, device, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        self.device = device

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1].to(self.device)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

class SFTDataLoaderLite:
    def __init__(self, B, device, process_rank, num_processes, split, master_process):
        self.B = B
        self.device = device
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}, "split must be 'train' or 'val'"

        data_root = "oasst_sft"
        shards = sorted([
            os.path.join(data_root, f) for f in os.listdir(data_root)
            if split in f
        ])
        assert shards, f"No shards found for split {split} in {data_root}"
        if master_process:
            print(f"found {len(shards)} OASST SFT shards for split {split}")
        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.examples = torch.load(self.shards[self.current_shard])
        self.current_position = self.B * self.process_rank

    def _pad_batch(self, examples):
        """
        Pads input_ids and labels to the max length in batch.
        """
        max_len = max(e['input_ids'].size(0) for e in examples)
        input_ids = torch.full((len(examples), max_len), fill_value=0, dtype=torch.long)
        labels = torch.full((len(examples), max_len), fill_value=-100, dtype=torch.long)
        for i, ex in enumerate(examples):
            length = ex['input_ids'].size(0)
            input_ids[i, :length] = ex['input_ids']
            labels[i, :length] = ex['labels']
        return input_ids.to(self.device), labels.to(self.device)

    def next_batch(self):
        B = self.B
        batch = []
        while len(batch) < B:
            if self.current_position >= len(self.examples):
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.examples = torch.load(self.shards[self.current_shard])
                self.current_position = self.B * self.process_rank

            idx = self.current_position
            step = self.num_processes
            while len(batch) < B and idx < len(self.examples):
                batch.append(self.examples[idx])
                idx += step
            self.current_position = idx

        return self._pad_batch(batch)

def safe_compile(model):
    try:
        compiled_model = torch.compile(model)
        # Try a dry-run forward pass to catch runtime errors (like TritonMissing)
        with torch.no_grad():
            compiled_model(torch.randn(4, 1024), torch.randn(4, 1024))
        return compiled_model
    except Exception as e:
        
        print("[torch.compile] Falling back to eager mode.")
        return torch.compile(model, backend="eager")

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
def get_lr(it, max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)