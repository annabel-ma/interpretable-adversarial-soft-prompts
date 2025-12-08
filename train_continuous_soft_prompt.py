#!/usr/bin/env python3
"""
Script to train a continuous soft prompt using the scaffold from all_exp_fixed_pez.ipynb
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)


@dataclass
class ExperimentConfig:
    model_name: str = "t5-large"
    gpt2_name: str = "gpt2"
    max_source_length: int = 256
    max_target_length: int = 4
    batch_size: int = 16
    lr: float = 1e-3
    num_epochs: int = 5
    prompt_length: int = 10
    lambda_grid: List[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./results/continuous_soft_prompt"
    seed: int = 42

    def __post_init__(self):
        if self.lambda_grid is None:
            self.lambda_grid = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


def preprocess_boolq(example, tokenizer: T5TokenizerFast, max_source_length: int, max_target_length: int):
    """
    Turn BoolQ into T5 inputs:
      "question: {question} passage: {passage}"
    Targets are "yes" or "no".
    """
    question = example["question"]
    passage = example["passage"]
    answer = "yes" if example["answer"] else "no"

    source = f"question: {question} passage: {passage}"
    target = answer

    model_inputs = tokenizer(
        source,
        truncation=True,
        padding="max_length",
        max_length=max_source_length,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target,
            truncation=True,
            padding="max_length",
            max_length=max_target_length,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_boolq_balanced(tokenizer: T5TokenizerFast, cfg: ExperimentConfig, seed: int = 42):
    """
    Load BoolQ, balance the train split on the raw dataset (True/False),
    then preprocess into T5-style inputs and wrap in DataLoaders.
    """
    raw = load_dataset("boolq")
    train_raw = raw["train"]
    val_raw   = raw["validation"]

    # --- Balance train: downsample majority label ---
    true_indices  = [i for i, ex in enumerate(train_raw) if ex["answer"]]
    false_indices = [i for i, ex in enumerate(train_raw) if not ex["answer"]]

    min_count = min(len(true_indices), len(false_indices))
    true_indices  = true_indices[:min_count]
    false_indices = false_indices[:min_count]

    balanced_indices = true_indices + false_indices
    # Optional but usually helpful: shuffle indices for randomness
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(balanced_indices), generator=rng).tolist()
    balanced_indices = [balanced_indices[i] for i in perm]

    train_balanced = train_raw.select(balanced_indices)

    print(
        f"Balanced BoolQ train: {len(train_balanced)} examples "
        f"({min_count} True, {min_count} False)"
    )

    # --- Preprocess to T5 format ---
    def preprocess_fn(ex):
        return preprocess_boolq(
            ex,
            tokenizer=tokenizer,
            max_source_length=cfg.max_source_length,
            max_target_length=cfg.max_target_length,
        )

    train_proc = train_balanced.map(preprocess_fn, batched=False)
    val_proc   = val_raw.map(preprocess_fn, batched=False)

    # Keep only the model fields and cast to torch
    cols = ["input_ids", "attention_mask", "labels"]
    train_proc.set_format(type="torch", columns=cols)
    val_proc.set_format(type="torch", columns=cols)

    train_dl = DataLoader(train_proc, batch_size=cfg.batch_size, shuffle=True)
    val_dl   = DataLoader(val_proc,   batch_size=cfg.batch_size, shuffle=False)

    return train_dl, val_dl


class T5ContinuousSoftPrompt(nn.Module):
    """
    Continuous soft prompts: learn a (prompt_length, d_model) tensor
    and prepend it as prefix embeddings to the encoder input.
    """

    def __init__(self, base: T5ForConditionalGeneration, prompt_length: int):
        super().__init__()
        self.t5 = base
        self.prompt_length = prompt_length

        d_model = base.encoder.embed_tokens.weight.shape[1]
        self.soft_prompt = nn.Parameter(
            torch.zeros(prompt_length, d_model)
        )
        nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Original token embeddings
        inputs_embeds = self.t5.encoder.embed_tokens(input_ids)

        # Broadcast prompt to batch
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Prepend to input sequence
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # Extend attention mask
        prompt_mask = torch.ones(batch_size, self.prompt_length, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs  # has .loss and .logits


@torch.no_grad()
def evaluate_accuracy_t5(model: nn.Module, dataloader: DataLoader,
                         tokenizer: T5TokenizerFast, device: str) -> Dict[str, float]:
    """
    BoolQ accuracy for T5-style models:
    - We look at the first decoder position's logits (position 0 in labels)
    - Compare scores for 'yes' vs 'no'
    
    Returns a dictionary with:
    - 'overall': overall accuracy
    - 'true_acc': accuracy on questions where answer is True (yes)
    - 'false_acc': accuracy on questions where answer is False (no)
    """

    model.eval()

    # Correct way to get the IDs for "yes" / "no" for T5
    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_id  = tokenizer("no",  add_special_tokens=False).input_ids[0]

    correct = 0
    total = 0
    
    # Separate tracking for True and False answers
    correct_true = 0
    total_true = 0
    correct_false = 0
    total_false = 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass: all your wrappers accept labels=...
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # logits: (B, T_out, V)
        logits = outputs.logits

        # First decoder step (the token that should be "yes" or "no")
        first_step_logits = logits[:, 0, :]  # (B, V)

        # Scores only for "yes" and "no"
        yes_scores = first_step_logits[:, yes_id]
        no_scores  = first_step_logits[:, no_id]

        # Predict yes if yes_score >= no_score else no
        pred_is_yes = (yes_scores >= no_scores)

        # Ground truth: first label token
        target_ids  = labels[:, 0]
        target_is_yes = (target_ids == yes_id)
        target_is_no  = (target_ids == no_id)

        # Correct if our yes/no prediction matches target
        correct_batch = (pred_is_yes & target_is_yes) | (~pred_is_yes & target_is_no)
        correct += correct_batch.sum().item()
        total   += target_ids.size(0)
        
        # Track accuracy separately for True and False answers
        # True answers (yes)
        true_mask = target_is_yes
        if true_mask.any():
            correct_true_batch = (pred_is_yes & target_is_yes)[true_mask]
            correct_true += correct_true_batch.sum().item()
            total_true += true_mask.sum().item()
        
        # False answers (no)
        false_mask = target_is_no
        if false_mask.any():
            correct_false_batch = (~pred_is_yes & target_is_no)[false_mask]
            correct_false += correct_false_batch.sum().item()
            total_false += false_mask.sum().item()

    overall_acc = correct / total if total > 0 else 0.0
    true_acc = correct_true / total_true if total_true > 0 else 0.0
    false_acc = correct_false / total_false if total_false > 0 else 0.0
    
    return {
        "overall": overall_acc,
        "true_acc": true_acc,
        "false_acc": false_acc
    }


def train_continuous_soft_prompt(
    cfg: ExperimentConfig,
    tokenizer: T5TokenizerFast,
    train_dl: DataLoader,
    val_dl: DataLoader,
    adversarial: bool,
) -> Dict[str, Any]:
    device = cfg.device
    base = T5ForConditionalGeneration.from_pretrained(cfg.model_name).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    model = T5ContinuousSoftPrompt(base, prompt_length=cfg.prompt_length).to(device)

    # Use lower learning rate for adversarial training to prevent explosion
    effective_lr = cfg.lr * 0.1 if adversarial else cfg.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.01)

    # --- label ids for flipping (yes/no) ---
    if adversarial:
        yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
        no_id  = tokenizer("no",  add_special_tokens=False).input_ids[0]
    # --------------------------------------#

    history = {
        "train_joint": [],
        "train_task": [],
        "val_loss": [],
        "val_acc": [],
        "val_acc_true": [],
        "val_acc_false": [],
        "prompt_norm": [],
    }

    for epoch in range(cfg.num_epochs):
        model.train()
        running_joint = 0.0
        running_task  = 0.0
        n_batches = 0

        for batch in train_dl:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # ---- flip labels during adversarial TRAINING ----
            if adversarial:
                labels_flipped = labels.clone()
                mask_yes = labels == yes_id
                mask_no  = labels == no_id
                labels_flipped[mask_yes] = no_id
                labels_flipped[mask_no]  = yes_id
                labels_for_loss = labels_flipped
            else:
                labels_for_loss = labels
            # -------------------------------------------------#

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_for_loss,
            )
            task_loss = outputs.loss
            joint_loss = task_loss  # no sign flip

            joint_loss.backward()
            # gradient clipping (tighter for adversarial)
            max_norm = 0.5 if adversarial else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            running_joint += joint_loss.item()
            running_task  += task_loss.item()
            n_batches += 1

        avg_train_joint = running_joint / max(1, n_batches)
        avg_train_task  = running_task  / max(1, n_batches)

        # ----------------- validation: TRUE labels -----------------
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,  # real yes/no labels
                )
                val_loss += outputs.loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        val_acc_dict = evaluate_accuracy_t5(model, val_dl, tokenizer, device)
        val_acc = val_acc_dict["overall"]
        val_acc_true = val_acc_dict["true_acc"]
        val_acc_false = val_acc_dict["false_acc"]

        prompt_norm = model.soft_prompt.norm().item()

        history["train_joint"].append(avg_train_joint)
        history["train_task"].append(avg_train_task)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_acc_true"].append(val_acc_true)
        history["val_acc_false"].append(val_acc_false)
        history["prompt_norm"].append(prompt_norm)

        print(
            f"[Continuous {'ADV' if adversarial else 'NON-ADV'}] "
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"joint={avg_train_joint:.4f} task={avg_train_task:.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} "
            f"(true={val_acc_true:.4f} false={val_acc_false:.4f}) "
            f"‖prompt‖={prompt_norm:.2f}"
        )

    return {"model": model, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train continuous soft prompt")
    parser.add_argument("--model_name", type=str, default="t5-large",
                        help="T5 model name (default: t5-large)")
    parser.add_argument("--prompt_length", type=int, default=10,
                        help="Length of soft prompt (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Use adversarial training (flip labels during training)")
    parser.add_argument("--output_dir", type=str, default="./results/continuous_soft_prompt",
                        help="Output directory for results (default: ./results/continuous_soft_prompt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Create config
    cfg = ExperimentConfig(
        model_name=args.model_name,
        prompt_length=args.prompt_length,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    if args.device:
        cfg.device = args.device
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    print("=" * 80)
    print("Continuous Soft Prompt Training")
    print("=" * 80)
    print(f"Model: {cfg.model_name}")
    print(f"Prompt Length: {cfg.prompt_length}")
    print(f"Learning Rate: {cfg.lr}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Adversarial: {args.adversarial}")
    print(f"Device: {cfg.device}")
    print(f"Output Dir: {cfg.output_dir}")
    print("=" * 80)
    
    # Load tokenizer and data
    print("\nLoading tokenizer and data...")
    tokenizer = T5TokenizerFast.from_pretrained(cfg.model_name)
    train_dl, val_dl = load_boolq_balanced(tokenizer, cfg, seed=cfg.seed)
    
    # Train
    print(f"\nTraining continuous soft prompt ({'ADVERSARIAL' if args.adversarial else 'NON-ADVERSARIAL'})...")
    result = train_continuous_soft_prompt(
        cfg, tokenizer, train_dl, val_dl, adversarial=args.adversarial
    )
    
    # Save results
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(
        cfg.output_dir,
        f"model_adversarial_{args.adversarial}_lr_{cfg.lr}_promptlen_{cfg.prompt_length}.pt"
    )
    torch.save(result["model"].state_dict(), model_path)
    print(f"\nSaved model to: {model_path}")
    
    # Save history
    history_path = os.path.join(
        cfg.output_dir,
        f"history_adversarial_{args.adversarial}_lr_{cfg.lr}_promptlen_{cfg.prompt_length}.json"
    )
    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)
    print(f"Saved history to: {history_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    best_val_acc = max(result["history"]["val_acc"])
    final_val_acc = result["history"]["val_acc"][-1]
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Best True Accuracy: {max(result['history']['val_acc_true']):.4f}")
    print(f"Best False Accuracy: {max(result['history']['val_acc_false']):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

