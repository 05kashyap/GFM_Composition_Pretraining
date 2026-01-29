"""
Training utilities for model training.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


def setup_device() -> torch.device:
    """Setup and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    output_dir: Path,
    filename: str = "checkpoint.pth",
) -> None:
    """
    Save a checkpoint.
    
    Args:
        state: State dict to save
        is_best: Whether this is the best model so far
        output_dir: Directory to save to
        filename: Checkpoint filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / filename
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = output_dir / "best_model.pth"
        shutil.copyfile(checkpoint_path, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[int, float]:
    """
    Load a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
    
    Returns:
        Tuple of (epoch, best_acc)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)
    
    return epoch, best_acc


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> list:
    """
    Computes the accuracy over the k top predictions.
    
    Args:
        output: Model output logits [B, C]
        target: Target labels [B]
        topk: Tuple of k values to compute accuracy for
    
    Returns:
        List of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res