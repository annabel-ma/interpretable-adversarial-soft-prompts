#!/usr/bin/env python3
"""
PEZ Single Training Script
Trains a single PEZ model with given lambda and learning rate.
"""

import argparse
import math
import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from torch.utils.data import DataLoader
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

    def __post_init__(self):
        if self.lambda_grid is None:
            self.lambda_grid = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


def preprocess_boolq(example, tokenizer: T5TokenizerFast, max_source_length: int, max_target_length: int):
    """Turn BoolQ into T5 inputs: "question: {question} passage: {passage}"."""
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
    """Load BoolQ, balance the train split, then preprocess into T5-style inputs."""
    raw = load_dataset("boolq")
    train_raw = raw["train"]
    val_raw   = raw["validation"]

    # Balance train: downsample majority label
    true_indices  = [i for i, ex in enumerate(train_raw) if ex["answer"]]
    false_indices = [i for i, ex in enumerate(train_raw) if not ex["answer"]]

    min_count = min(len(true_indices), len(false_indices))
    true_indices  = true_indices[:min_count]
    false_indices = false_indices[:min_count]

    balanced_indices = true_indices + false_indices
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(balanced_indices), generator=rng).tolist()
    balanced_indices = [balanced_indices[i] for i in perm]

    train_balanced = train_raw.select(balanced_indices)

    print(
        f"Balanced BoolQ train: {len(train_balanced)} examples "
        f"({min_count} True, {min_count} False)"
    )

    def preprocess_fn(ex):
        return preprocess_boolq(
            ex,
            tokenizer=tokenizer,
            max_source_length=cfg.max_source_length,
            max_target_length=cfg.max_target_length,
        )

    train_proc = train_balanced.map(preprocess_fn, batched=False)
    val_proc   = val_raw.map(preprocess_fn, batched=False)

    cols = ["input_ids", "attention_mask", "labels"]
    train_proc.set_format(type="torch", columns=cols)
    val_proc.set_format(type="torch", columns=cols)

    train_dl = DataLoader(train_proc, batch_size=cfg.batch_size, shuffle=True)
    val_dl   = DataLoader(val_proc,   batch_size=cfg.batch_size, shuffle=False)

    return train_dl, val_dl


class T5PEZPrompt(nn.Module):
    """
    PEZ-style one-hot prompt over the T5 vocabulary.

    - self.prompt is (L, V) with rows ~ one-hot
    - FORWARD: treat self.prompt as continuous, embed via matrix multiply
               prompt_embeds = prompt @ embed_tokens.weight
      This keeps the computation differentiable w.r.t. self.prompt.
    - DISCRETIZATION: use argmax *outside* the forward (for decoding / GPT-2).
    """

    def __init__(self, base: T5ForConditionalGeneration, prompt_length: int):
        super().__init__()
        self.t5 = base
        self.vocab_size = base.encoder.embed_tokens.weight.shape[0]
        self.prompt_length = prompt_length

        # Initialize as random one-hot rows
        init_ids = torch.randint(0, self.vocab_size, (prompt_length,))
        prompt = torch.nn.functional.one_hot(init_ids, num_classes=self.vocab_size).float()
        self.prompt = nn.Parameter(prompt)  # (L, V), requires_grad=True by default

    # ---- helper used ONLY for decoding / perplexity, NOT in forward ----
    def get_prompt_token_ids(self) -> torch.Tensor:
        return self.prompt.argmax(dim=-1)  # (L,)

    def decode_prompt(self, tokenizer: T5TokenizerFast) -> str:
        token_ids = self.get_prompt_token_ids().tolist()
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_projected_prompt_embeds(self, batch_size: int) -> torch.Tensor:
        """
        PEZ projection: Proj_E(P) = embed(argmax(P))
        Returns projected prompt embeddings for use in forward pass.
        This is differentiable via straight-through estimator.
        """
        # Get token IDs via argmax (discrete projection)
        token_ids = self.get_prompt_token_ids()  # (L,)
        # Embed the projected tokens
        embed_matrix = self.t5.encoder.embed_tokens.weight  # (V, d_model)
        prompt_embeds = embed_matrix[token_ids]  # (L, d_model)
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return prompt_embeds

    # ---- differentiable forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        use_projection: bool = False,
    ):
        """
        Embed prompt by multiplying the (L, V) prompt matrix with the
        (V, d_model) embedding matrix. This is linear in self.prompt,
        so gradients flow into self.prompt.
        
        If use_projection=True, uses PEZ projection: Proj_E(P) = embed(argmax(P))
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # 1) Embed the prompt: either continuous or projected
        if use_projection:
            # PEZ: use projected embeddings (discrete tokens)
            prompt_embeds = self.get_projected_prompt_embeds(batch_size)
        else:
            # Original: continuous embedding
            embed_matrix = self.t5.encoder.embed_tokens.weight  # (V, d_model)
            prompt_embeds = self.prompt @ embed_matrix          # (L, d_model)
            prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        # 2) Embed the original input tokens
        inputs_embeds = self.t5.encoder.embed_tokens(input_ids)

        # 3) Prepend prompt embeddings
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # 4) Extend attention mask
        prompt_mask = torch.ones(
            batch_size,
            self.prompt_length,
            device=device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # 5) Standard T5 forward
        outputs = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs  # has .loss and .logits


def compute_prompt_ppl_loss_from_text(
    gpt2_model: GPT2LMHeadModel,
    gpt2_tokenizer: GPT2TokenizerFast,
    prompt_text: str,
    device: str,
) -> torch.Tensor:
    """Take decoded prompt text, feed to GPT-2, compute LM loss."""
    if not prompt_text or not prompt_text.strip():
        return torch.tensor(0.0, device=device)

    enc = gpt2_tokenizer(prompt_text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)

    if input_ids.numel() == 0:
        return torch.tensor(0.0, device=device)

    labels = input_ids.clone()
    with torch.no_grad():
        outputs = gpt2_model(input_ids=input_ids, labels=labels)
    return outputs.loss


@torch.no_grad()
def evaluate_accuracy_t5(model: nn.Module, dataloader: DataLoader,
                         tokenizer: T5TokenizerFast, device: str) -> Dict[str, float]:
    """BoolQ accuracy for T5-style models."""
    model.eval()

    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_id  = tokenizer("no",  add_special_tokens=False).input_ids[0]

    correct = 0
    total = 0
    correct_true = 0
    total_true = 0
    correct_false = 0
    total_false = 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits

        first_step_logits = logits[:, 0, :]
        yes_scores = first_step_logits[:, yes_id]
        no_scores  = first_step_logits[:, no_id]
        pred_is_yes = (yes_scores >= no_scores)

        target_ids  = labels[:, 0]
        target_is_yes = (target_ids == yes_id)
        target_is_no  = (target_ids == no_id)

        correct_batch = (pred_is_yes & target_is_yes) | (~pred_is_yes & target_is_no)
        correct += correct_batch.sum().item()
        total   += target_ids.size(0)
        
        true_mask = target_is_yes
        if true_mask.any():
            correct_true_batch = (pred_is_yes & target_is_yes)[true_mask]
            correct_true += correct_true_batch.sum().item()
            total_true += true_mask.sum().item()
        
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


def train_pez(
    cfg,
    tokenizer: T5TokenizerFast,
    gpt2_model: GPT2LMHeadModel,
    gpt2_tokenizer: GPT2TokenizerFast,
    train_dl: DataLoader,
    val_dl: DataLoader,
    lambda_ppl: float,
    adversarial: bool,
    log_every: int = 50,
):
    device = cfg.device
    base = T5ForConditionalGeneration.from_pretrained(cfg.model_name).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    model = T5PEZPrompt(base, prompt_length=cfg.prompt_length).to(device)

    # --- label ids for flipping (yes/no) ---
    if adversarial:
        yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
        no_id  = tokenizer("no",  add_special_tokens=False).input_ids[0]
    # --------------------------------------#

    history = {
        "lambda_ppl": lambda_ppl,
        "train_joint": [],
        "train_task": [],
        "train_ppl_loss": [],
        "train_ppl_ppx": [],
        "val_loss": [],
        "val_acc": [],
        "val_acc_true": [],
        "val_acc_false": [],
        "prompt_ppl_ppx": [],
    }

    for epoch in range(cfg.num_epochs):
        model.train()
        running_joint = 0.0
        running_task = 0.0
        running_ppl  = 0.0
        running_ppl_ppx = 0.0
        n_batches = 0

        empty_ppl_calls = 0
        nonempty_ppl_calls = 0

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

            model.prompt.grad = None

            # ---- PEZ Algorithm: Forward with projected prompt ----
            # According to PEZ paper: P' = Proj_E(P), then compute L(B(P', X), Y)
            # We use projected embeddings (discrete tokens) in forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_for_loss,
                use_projection=True,  # Use projected embeddings: Proj_E(P)
            )
            task_loss = outputs.loss

            # ---- perplexity term ----
            if lambda_ppl > 0.0:
                prompt_ids = model.get_prompt_token_ids()
                prompt_text = tokenizer.decode(
                    prompt_ids.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()

                if not prompt_text:
                    ppl_loss = torch.tensor(0.0, device=device)
                    empty_ppl_calls += 1
                else:
                    nonempty_ppl_calls += 1
                    ppl_loss = compute_prompt_ppl_loss_from_text(
                        gpt2_model, gpt2_tokenizer, prompt_text, device=device
                    )
                    if torch.isnan(ppl_loss) or torch.isinf(ppl_loss):
                        ppl_loss = torch.tensor(0.0, device=device)
            else:
                ppl_loss = torch.tensor(0.0, device=device)

            joint_loss = task_loss + lambda_ppl * ppl_loss
            
            # ---- PEZ Algorithm: Compute gradient w.r.t. projected prompt ----
            # Since projection (argmax) is non-differentiable, we need to handle gradients
            # We use continuous prompt for gradient computation (straight-through estimator)
            # Forward used projected, but for backward we need gradients w.r.t. continuous
            
            # Re-run forward with continuous prompt to get gradient flow
            outputs_cont = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_for_loss,
                use_projection=False,  # Use continuous for gradient computation
            )
            task_loss_cont = outputs_cont.loss
            joint_loss_cont = task_loss_cont + lambda_ppl * ppl_loss
            joint_loss_cont.backward()
            
            # ---- PEZ Algorithm: Update continuous prompt ----
            # P = P - γ * ∇_P' L, where P' = Proj_E(P)
            # Using straight-through: ∇_P' L ≈ ∇_P L (gradient w.r.t. continuous approximates gradient w.r.t. projected)
            with torch.no_grad():
                if model.prompt.grad is not None:
                    # Update: P = P - lr * ∇_P L
                    model.prompt.data = model.prompt.data - cfg.lr * model.prompt.grad
                    model.prompt.grad.zero_()
            # ---------------------------------------------

            running_joint += joint_loss.item()
            running_task  += task_loss.item()
            running_ppl   += ppl_loss.item()
            if lambda_ppl > 0.0:
                running_ppl_ppx += math.exp(ppl_loss.item())
            n_batches += 1

            if (n_batches % log_every) == 0:
                avg_joint_so_far = running_joint / n_batches
                avg_task_so_far  = running_task  / n_batches
                avg_ppl_so_far   = running_ppl   / n_batches
                avg_ppl_ppx_so_far = (
                    running_ppl_ppx / n_batches if lambda_ppl > 0.0 else 0.0
                )
                print(
                    f"[PEZ λ={lambda_ppl} {'ADV' if adversarial else 'NON-ADV'}] "
                    f"Epoch {epoch+1}/{cfg.num_epochs}, "
                    f"batch {n_batches} | "
                    f"joint={avg_joint_so_far:.4f} "
                    f"task={avg_task_so_far:.4f} "
                    f"ppl_loss={avg_ppl_so_far:.4f} "
                    f"ppl={avg_ppl_ppx_so_far:.2f}"
                )

        # ---- end-of-epoch aggregation ----
        avg_joint = running_joint / max(1, n_batches)
        avg_task  = running_task  / max(1, n_batches)
        avg_ppl   = running_ppl   / max(1, n_batches)
        avg_ppl_ppx = running_ppl_ppx / max(1, n_batches) if lambda_ppl > 0.0 else 0.0

        history["train_joint"].append(avg_joint)
        history["train_task"].append(avg_task)
        history["train_ppl_loss"].append(avg_ppl)
        history["train_ppl_ppx"].append(avg_ppl_ppx)

        # ---- validation: TRUE labels ----
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
                    labels=labels,   # real yes/no labels here
                )
                val_loss += outputs.loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(1, val_batches)
        val_acc_dict = evaluate_accuracy_t5(model, val_dl, tokenizer, device)
        val_acc = val_acc_dict["overall"]
        val_acc_true = val_acc_dict["true_acc"]
        val_acc_false = val_acc_dict["false_acc"]

        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_acc_true"].append(val_acc_true)
        history["val_acc_false"].append(val_acc_false)

        # ---- prompt perplexity once per epoch ----
        if lambda_ppl > 0.0:
            prompt_text_epoch = model.decode_prompt(tokenizer).strip()
            if prompt_text_epoch:
                ppl_loss_epoch = compute_prompt_ppl_loss_from_text(
                    gpt2_model, gpt2_tokenizer, prompt_text_epoch, device=device
                )
                prompt_ppx = math.exp(ppl_loss_epoch.item())
            else:
                prompt_ppx = float("nan")
        else:
            prompt_ppx = 0.0
        history["prompt_ppl_ppx"].append(prompt_ppx)

        decoded_prompt = model.decode_prompt(tokenizer)
        print(
            f"[PEZ λ={lambda_ppl} {'ADV' if adversarial else 'NON-ADV'}] "
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"joint={avg_joint:.4f} task={avg_task:.4f} "
            f"ppl_loss={avg_ppl:.4f} ppl={avg_ppl_ppx:.2f} "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} "
            f"(true={val_acc_true:.4f} false={val_acc_false:.4f}) "
            f"prompt_ppl={prompt_ppx:.2f}\n"
            f"Prompt: {decoded_prompt}"
        )

    return {"model": model, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train PEZ model with given lambda and learning rate")
    parser.add_argument("--lambda_ppl", type=float, required=True, help="Lambda for perplexity loss")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--adversarial", action="store_true", help="Use adversarial training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--prompt_length", type=int, default=10, help="Prompt length")
    parser.add_argument("--model_name", type=str, default="t5-large", help="T5 model name")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches")
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize config
    cfg = ExperimentConfig()
    cfg.lr = args.lr
    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.prompt_length = args.prompt_length
    cfg.model_name = args.model_name
    device = cfg.device
    
    print(f"Using device: {device}")
    print(f"Lambda: {args.lambda_ppl}, LR: {args.lr}, Adversarial: {args.adversarial}")
    
    # Load tokenizer
    print("\nLoading T5 tokenizer...")
    tokenizer = T5TokenizerFast.from_pretrained(cfg.model_name)
    
    # Load GPT-2
    print("\nLoading GPT-2 for prompt perplexity...")
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(cfg.gpt2_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(cfg.gpt2_name).to(device)
    gpt2_model.eval()
    for p in gpt2_model.parameters():
        p.requires_grad = False
    
    # Load data
    print("\nLoading BoolQ dataset...")
    train_dl, val_dl = load_boolq_balanced(tokenizer, cfg)
    
    # Train
    result = train_pez(
        cfg,
        tokenizer,
        gpt2_model,
        gpt2_tokenizer,
        train_dl,
        val_dl,
        lambda_ppl=args.lambda_ppl,
        adversarial=args.adversarial,
        log_every=args.log_every,
    )
    
    # Save results
    model = result["model"]
    history = result["history"]
    
    # Create subdirectory based on adversarial flag
    adv_subdir = "adversarial_true" if args.adversarial else "adversarial_false"
    output_subdir = os.path.join(args.output_dir, adv_subdir)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Create filenames (without adversarial suffix since it's in the subdir)
    # Include lambda, lr, and prompt_length to ensure unique filenames
    model_filename = f"model_lambda_{args.lambda_ppl}_lr_{args.lr}_promptlen_{args.prompt_length}.pt"
    history_filename = f"history_lambda_{args.lambda_ppl}_lr_{args.lr}_promptlen_{args.prompt_length}.json"
    
    torch.save(model.state_dict(), os.path.join(output_subdir, model_filename))
    with open(os.path.join(output_subdir, history_filename), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to {output_subdir}")
    print(f"  Model: {model_filename}")
    print(f"  History: {history_filename}")


if __name__ == "__main__":
    main()

