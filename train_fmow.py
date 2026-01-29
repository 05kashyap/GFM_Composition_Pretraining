"""
Training script for fMoW dataset with DynamicVis architecture.
Streams data directly from AWS S3 without downloading the full dataset.
"""

import os
import sys
import argparse
from pathlib import Path

# Add architectures to path
sys.path.insert(0, str(Path(__file__).parent / "architectures" / "DynamicVis"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

from datasets.fmow_s3_dataset import FMoWS3Dataset, get_fmow_transforms
from models.dynamicvis_classifier import build_dynamicvis_classifier
from utils.training_utils import (
    setup_device,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    accuracy,
)
from configs.fmow_config import FMoWConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train DynamicVis on fMoW dataset")
    parser.add_argument("--config", type=str, default="configs/fmow_config.py", help="Config file path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--train-steps", type=int, default=None, help="Limit training steps per epoch (for debugging)")
    parser.add_argument("--val-steps", type=int, default=None, help="Limit validation steps (for debugging)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, max_steps=None):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    
    train_iterator = iter(train_loader)
    num_steps = max_steps if max_steps else len(train_loader)
    
    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch} [Train]")
    
    for step in pbar:
        try:
            images, labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            images, labels = next(train_iterator)
        
        # Filter out invalid samples (label == -1)
        valid_mask = labels != -1
        if not valid_mask.any():
            continue
            
        images = images[valid_mask].to(device)
        labels = labels[valid_mask].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        
        pbar.set_postfix({
            "loss": f"{losses.avg:.4f}",
            "acc@1": f"{top1.avg:.2f}%",
            "acc@5": f"{top5.avg:.2f}%"
        })
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, max_steps=None):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter("Loss")
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    
    num_steps = max_steps if max_steps else len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), desc="Validating", total=num_steps)
        
        for step, (images, labels) in pbar:
            if max_steps and step >= max_steps:
                break
            
            # Filter out invalid samples
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
                
            images = images[valid_mask].to(device)
            labels = labels[valid_mask].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            pbar.set_postfix({
                "loss": f"{losses.avg:.4f}",
                "acc@1": f"{top1.avg:.2f}%"
            })
    
    return losses.avg, top1.avg, top5.avg


def main():
    args = parse_args()
    
    # Load environment variables (AWS credentials)
    load_dotenv()
    
    # Verify AWS credentials are set
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("WARNING: AWS credentials not found in environment. Using unsigned requests.")
    
    # Setup device
    device = setup_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = FMoWConfig()
    
    # Override config with command line args
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_workers = args.num_workers
    
    # Get transforms
    train_transform, val_transform = get_fmow_transforms(config.image_size)
    
    # Create datasets
    print("\n--- Creating Datasets ---")
    train_dataset = FMoWS3Dataset(
        bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix,
        manifest_key=config.manifest_key,
        local_manifest=config.local_manifest,
        split="train",
        transform=train_transform,
    )
    
    val_dataset = FMoWS3Dataset(
        bucket=config.s3_bucket,
        s3_prefix=config.s3_prefix,
        manifest_key=config.manifest_key,
        local_manifest=config.local_manifest,
        split="val",
        transform=val_transform,
    )
    
    # Update num_classes from dataset
    config.num_classes = len(train_dataset.class_to_idx)
    print(f"Number of classes: {config.num_classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Build model
    print("\n--- Building Model ---")
    model = build_dynamicvis_classifier(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        model_type=config.model_type,
        img_size=config.image_size,
    )
    model = model.to(device)
    print(f"Model: {config.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=config.learning_rate * 0.01,
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    # Training loop
    print(f"\n--- Starting Training for {args.epochs} epochs ---")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_steps=args.train_steps,
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device,
            max_steps=args.val_steps,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc@1: {train_acc1:.2f}%, Acc@5: {train_acc5:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)
        
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
                "config": config.__dict__,
            },
            is_best,
            output_dir,
        )
    
    print(f"\n--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()