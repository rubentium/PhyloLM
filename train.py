import os
import math
import time
import logging
import argparse

import wandb
import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np

from data import Tokenizer
from model.memmap_data import create_memmap_dataloaders
from model.model import PhyloLM
from model.sparse_attention import _MASK_POOL_SIZE

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# number of pre-generated masks to pool for random selection during training,
# the memory they take up is a rounding error!

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--memmap_dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "..", "LG_GC_memmaps"),
    )
    parser.add_argument("--prefetch", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1) # you will run out of GPU VRAM if you increase this

    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--att_type", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--num_random_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--permute_pairs", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--criterion", type=str, default="mse", choices=["mre", "mse", "mae"])

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=48)

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=2499)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--wandb_project", type=str, default="PhyloLM")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")

    parser.add_argument("--seed", type=int, default=1004)

    return parser.parse_args()


def MRE(preds, distances):
    eps = 1e-8
    return torch.mean(torch.abs((preds - distances) / (distances + eps)))


def compute_metrics(preds, distances):
    return {
        "mse": nn.functional.mse_loss(preds, distances),
        "mae": nn.functional.l1_loss(preds, distances),
        "mre": MRE(preds, distances),
    }


def build_optimizer(model, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args, steps_per_epoch):
    total_steps = args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return total_steps, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("global_step", 0), ckpt.get("epoch", 0)


def save_checkpoint(path, model, optimizer, scheduler, global_step, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
    }, path)


@torch.no_grad()
def run_validation(model, val_samples, criterion, device, use_wandb, global_step, full_eval=False):
    model.eval()
    val_metrics = {"mse": 0.0, "mae": 0.0, "mre": 0.0}
    num_samples = val_samples.num_samples if full_eval else 300
    for _ in range(num_samples): # take 300 samples from val set
        alignment, distances = next(val_samples)
        alignment = alignment.to(device, non_blocking=True)
        distances = distances.to(device, non_blocking=True)
        indices = np.random.randint(0, _MASK_POOL_SIZE, size=model.num_blocks) if model.att_type == "sparse" else None

        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(alignment, sparse_indices=indices)
            pad = model.pair_padding  # (0, 0, top, bottom)
            top, bottom = pad[2].item(), pad[3].item()
            if top > 0 or bottom > 0:
                preds = preds[:, top: preds.size(1) - bottom]
            metrics = compute_metrics(preds, distances)
            for name, value in metrics.items():
                val_metrics[name] += value.item()

    val_metrics = {name: value / num_samples for name, value in val_metrics.items()}
    val_loss = val_metrics[criterion]
    if use_wandb:
        wandb.log({
            "val_loss": val_loss,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_mre": val_metrics["mre"],
        }, step=global_step)
    model.train()
    return val_metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    tokenizer = Tokenizer()
    train_iter, val_iter = create_memmap_dataloaders(
        memmap_dir=args.memmap_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        prefetch=args.prefetch,
    )
    
    num_rows = train_iter.num_rows
    num_cols = train_iter.num_cols

    model = PhyloLM(
        num_rows=num_rows,
        num_cols=num_cols,
        num_blocks=args.num_blocks,
        h_dim=args.h_dim,
        num_heads=args.num_heads,
        vocab_size=len(tokenizer),
        dropout=args.dropout,
        att_type=args.att_type,
        num_random_blocks=args.num_random_blocks,
    ).to(device, dtype=torch.bfloat16)

    optimizer = build_optimizer(model, args)
    steps_per_epoch = train_iter.num_samples // int(args.batch_size*args.grad_accum_steps) if hasattr(train_iter, 'num_samples') else 1000
    total_steps, scheduler = build_scheduler(optimizer, args, steps_per_epoch=steps_per_epoch)
    criterion = args.criterion if args.criterion else "mre"

    global_step, start_epoch = 0, 0
    if args.resume is not None:
        global_step, start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train_samples = iter(train_iter)
    val_samples = iter(val_iter)
    min_val_logged = float("inf")
    
    print(f"Dataset info: train samples={train_iter.num_samples}, val samples={val_iter.num_samples}")
    print(f"Model info: num_blocks={args.num_blocks}, h_dim={args.h_dim}, num_heads={args.num_heads}, dropout={args.dropout}")
    print(f"Training info: {args.epochs} epochs ({total_steps} steps) with optimizer={args.optimizer}, lr={args.lr}, weight_decay={args.weight_decay}")
    if args.att_type == "dense":
        print(summary(model, input_size=(1, num_rows, num_cols), dtypes=[torch.long], device=str(device)))

    model = torch.compile(model)
    print("Starting training...")
    while global_step < total_steps:
        train_metrics = {"mse": 0.0, "mae": 0.0, "mre": 0.0}
        time_start = time.time()
        for _ in range(args.grad_accum_steps or 1):
            alignment, distances = next(train_samples)
            alignment, distances = alignment.to(device, non_blocking=True), distances.to(device, non_blocking=True)
            indices = np.random.randint(0, _MASK_POOL_SIZE, size=args.num_blocks) if args.att_type == "sparse" else None
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                random_perm = None
                valid_perm = None
                pad = model.pair_padding  # (0, 0, top, bottom)
                top, bottom = pad[2].item(), pad[3].item()

                # if padding is non-zero permute valud rows
                if args.permute_pairs:
                    num_valid = model.num_pairs - top - bottom
                    valid_perm = torch.randperm(num_valid, device=device)
                    if top > 0 or bottom > 0:
                        # keep padding rows fixed; only shuffle the valid middle rows
                        random_perm = torch.cat([
                            torch.arange(top, device=device),
                            valid_perm + top,
                            torch.arange(model.num_pairs - bottom, model.num_pairs, device=device),
                        ])
                    else:
                        random_perm = valid_perm

                preds = model(alignment, sparse_indices=indices, random_perm=random_perm)
                # strip padding from preds
                if top > 0 or bottom > 0:
                    preds = preds[:, top: preds.size(1) - bottom]

                # permute target using the same valid-row permutation
                if args.permute_pairs:
                    distances = distances.index_select(1, valid_perm)

                metrics = compute_metrics(preds, distances)
                for name, value in metrics.items():
                    train_metrics[name] += value.item() / (args.grad_accum_steps or 1)

                loss = metrics[criterion] / (args.grad_accum_steps or 1)
                loss.backward()

        total_time = time.time() - time_start
        avg_time = int((total_time / (args.grad_accum_steps or 1)) * 1000)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        global_step += 1

        if args.use_wandb:
            wandb.log({
                "train_loss": train_metrics[criterion],
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "train_mre": train_metrics["mre"],
                "lr": scheduler.get_last_lr()[0],
            }, step=global_step)
        
        if global_step % args.log_every == 0 and global_step > 0:
            val_metrics = run_validation(model, val_samples, criterion, device, args.use_wandb, global_step)
            val_loss = val_metrics[criterion]
            if val_loss > min_val_logged+0.15 and global_step > 2500:
                break
            
            min_val_logged = min(min_val_logged, val_loss)
            
            print(
                f"step: {global_step}, train_loss: {train_metrics[criterion]:.4f}, val_loss: {val_loss:.4f}, "
                f"train_mse: {train_metrics['mse']:.4f}, train_mae: {train_metrics['mae']:.4f}, "
                f"train_mre: {train_metrics['mre']:.4f}, val_mse: {val_metrics['mse']:.4f}, "
                f"val_mae: {val_metrics['mae']:.4f}, val_mre: {val_metrics['mre']:.4f}, time: {avg_time:.2f}ms"
            )
        
        if global_step % 5000 == 0 and global_step > 0:
            save_checkpoint(os.path.join(args.checkpoint_dir, f"checkpoint_{global_step}.pt"), model, optimizer, scheduler, global_step, start_epoch)

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    full_val_metrics = run_validation(model, val_samples, criterion, device, args.use_wandb, global_step, full_eval=True)
    full_val_loss = full_val_metrics["mre"]
    save_checkpoint(os.path.join(args.checkpoint_dir, f"final_checkpoint_{time_stamp}.pt"), model, optimizer, scheduler, global_step, start_epoch)
    if args.use_wandb:
        wandb.log({
            "total_val_loss": full_val_loss,
            "total_val_mse": full_val_metrics["mse"],
            "total_val_mae": full_val_metrics["mae"],
            "total_val_mre": full_val_metrics["mre"],
        }, step=global_step)
    print(f"Training complete. Total validation loss: {full_val_loss:.4f}")
    print(f"Final model checkpoint saved to {os.path.join(args.checkpoint_dir, f'final_checkpoint_{time_stamp}.pt')}")

if __name__ == "__main__":
    main()
