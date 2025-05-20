import os
import time
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils import SFTDataLoaderLite, safe_compile
from guru_model import Guru
from math import ceil
import tiktoken


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
# --------------------- DDP Setup ---------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), 'CUDA is required for DDP'
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')

torch.manual_seed(1337 + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337 + seed_offset)

# --------------------- Config ---------------------
total_number_of_examples = 23993
B = 4 if ddp_world_size == 1 else 64
grad_accum_steps = 128
steps_per_epoch = ceil(total_number_of_examples/( B * grad_accum_steps))
max_steps = steps_per_epoch * 4
eval_interval = 20
log_dir = "log_sft"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# --------------------- Data ---------------------
train_loader = SFTDataLoaderLite(B, device, ddp_rank, ddp_world_size, "train", master_process)
val_loader = SFTDataLoaderLite(B, device, ddp_rank, ddp_world_size, "val", master_process)

# --------------------- Load Latest Pretrained Checkpoint ---------------------
def get_latest_checkpoint(log_dir="log"):
    checkpoints = [
        f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")
    ]
    if not checkpoints:
        raise FileNotFoundError("No pretrained checkpoints found in 'log/'")

    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return os.path.join(log_dir, checkpoints[-1])

ckpt_path = get_latest_checkpoint()
print(f'Loading checkpoint: {ckpt_path}')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
torch.set_float32_matmul_precision('high')
model = Guru(checkpoint['config'])

clean_state_dict = lambda sd: {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
state_dict = clean_state_dict(checkpoint['model'])
model.load_state_dict(state_dict)

# --------------------- Wrap/Compile ---------------------
model.to(device)
model = safe_compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# --------------------- Optimizer with Fixed LR ---------------------
fixed_lr = 3e-5
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=fixed_lr, device=device)

# --------------------- Training Loop ---------------------
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    if step % eval_interval == 0 or step == max_steps - 1:
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            val_loss_steps = 10
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, val_loss = model(x, y)
                val_loss_total += val_loss.item()
        val_loss_avg = val_loss_total / val_loss_steps
        if master_process:
            print(f"[val] step {step:4d} | val loss: {val_loss_avg:.6f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_avg:.6f}\n")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step
            }
            torch.save(checkpoint, os.path.join(log_dir, f"model_{step:05d}.pt"))

            prompt_text = "<|user|> What are the differences between a CPU and a GPU?\n<|assistant|>"
            
            num_return_sequences = 1
            max_length = 32
            tokens = enc.encode(prompt_text, allowed_special={"<|user|>", "<|assistant|>"})
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")
            
    model.train()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = fixed_lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    if ddp:
        torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)

    t1 = time.time()
    if master_process:
        examples_per_sec = total_number_of_examples / (t1 - t0)
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | dt: {(t1-t0):.4f} |",
              f"lr: {fixed_lr:.4e} | ex/sec: {examples_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
