#!/usr/bin/env python3
"""
Evaluate Hard (Projected) Prompts vs Soft Prompts
Loads final checkpoints, projects them to discrete tokens, and evaluates validation accuracy.
For each lambda: selects best performance for non-adversarial, worst for adversarial.
Writes results to CSV.
"""

import json
import os
import glob
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
)
import pandas as pd
import argparse


class T5PEZPrompt(nn.Module):
    """PEZ-style one-hot prompt over the T5 vocabulary."""
    
    def __init__(self, base: T5ForConditionalGeneration, prompt_length: int):
        super().__init__()
        self.t5 = base
        self.vocab_size = base.encoder.embed_tokens.weight.shape[0]
        self.prompt_length = prompt_length
        
        # Initialize as random one-hot rows
        init_ids = torch.randint(0, self.vocab_size, (prompt_length,))
        prompt = torch.nn.functional.one_hot(init_ids, num_classes=self.vocab_size).float()
        self.prompt = nn.Parameter(prompt)
    
    def get_prompt_token_ids(self) -> torch.Tensor:
        return self.prompt.argmax(dim=-1)
    
    def get_projected_prompt_embeds(self, batch_size: int) -> torch.Tensor:
        """PEZ projection: Proj_E(P) = embed(argmax(P))"""
        token_ids = self.get_prompt_token_ids()
        embed_matrix = self.t5.encoder.embed_tokens.weight
        prompt_embeds = embed_matrix[token_ids]
        prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return prompt_embeds
    
    def forward(self, input_ids, attention_mask, labels, use_projection=False):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        if use_projection:
            prompt_embeds = self.get_projected_prompt_embeds(batch_size)
        else:
            embed_matrix = self.t5.encoder.embed_tokens.weight
            prompt_embeds = self.prompt @ embed_matrix
            prompt_embeds = prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        inputs_embeds = self.t5.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
        prompt_mask = torch.ones(batch_size, self.prompt_length, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        outputs = self.t5(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs


def preprocess_boolq(example, tokenizer, max_source_length=256, max_target_length=4):
    question = example["question"]
    passage = example["passage"]
    answer = "yes" if example["answer"] else "no"
    
    source = f"question: {question} passage: {passage}"
    target = answer
    
    model_inputs = tokenizer(source, truncation=True, padding="max_length", max_length=max_source_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, truncation=True, padding="max_length", max_length=max_target_length)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


@torch.no_grad()
def evaluate_accuracy_hard_prompt(model, dataloader, tokenizer, device):
    """Evaluate accuracy using hard (projected) prompts."""
    model.eval()
    
    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("no", add_special_tokens=False).input_ids[0]
    
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Use projected (hard) prompts
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_projection=True,  # Use hard prompts
        )
        logits = outputs.logits
        
        first_step_logits = logits[:, 0, :]
        yes_scores = first_step_logits[:, yes_id]
        no_scores = first_step_logits[:, no_id]
        pred_is_yes = (yes_scores >= no_scores)
        
        target_ids = labels[:, 0]
        target_is_yes = (target_ids == yes_id)
        target_is_no = (target_ids == no_id)
        
        correct_batch = (pred_is_yes & target_is_yes) | (~pred_is_yes & target_is_no)
        correct += correct_batch.sum().item()
        total += target_ids.size(0)
    
    return correct / total if total > 0 else 0.0


def load_all_histories(directory):
    """Load all history JSON files from a directory."""
    histories = {}
    
    # Find all history files
    history_files = glob.glob(os.path.join(directory, "history_*.json"))
    
    for filepath in history_files:
        filename = os.path.basename(filepath)
        # Try new format first (with promptlen)
        match = re.match(r"history_lambda_([0-9.]+)_lr_([0-9.e-]+)_promptlen_([0-9]+)\.json", filename)
        if match:
            lambda_val = float(match.group(1))
            lr_val = float(match.group(2))
            prompt_length = int(match.group(3))
            
            with open(filepath, 'r') as f:
                history = json.load(f)
            
            key = (lambda_val, lr_val, prompt_length)
            histories[key] = history
        else:
            # Try old format (without promptlen) for backward compatibility
            match_old = re.match(r"history_lambda_([0-9.]+)_lr_([0-9.e-]+)\.json", filename)
            if match_old:
                lambda_val = float(match_old.group(1))
                lr_val = float(match_old.group(2))
                
                with open(filepath, 'r') as f:
                    history = json.load(f)
                
                prompt_length = history.get('prompt_length', 10)
                key = (lambda_val, lr_val, prompt_length)
                histories[key] = history
    
    return histories


def main():
    parser = argparse.ArgumentParser(description="Evaluate hard (projected) prompts vs soft prompts")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Base results directory (should contain adversarial_false and adversarial_true subdirs)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV file path (default: results_dir/hard_vs_soft_prompts.csv)")
    parser.add_argument("--model_name", type=str, default="t5-large",
                        help="T5 model name")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available, else cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Setup directories
    non_adv_dir = os.path.join(args.results_dir, "adversarial_false")
    adv_dir = os.path.join(args.results_dir, "adversarial_true")
    
    if not os.path.exists(non_adv_dir) and not os.path.exists(adv_dir):
        print(f"Error: Neither {non_adv_dir} nor {adv_dir} exists")
        return
    
    # Load histories
    print("\nLoading histories...")
    non_adv_histories = {}
    if os.path.exists(non_adv_dir):
        non_adv_histories = load_all_histories(non_adv_dir)
        print(f"Loaded {len(non_adv_histories)} non-adversarial histories")
    
    adv_histories = {}
    if os.path.exists(adv_dir):
        adv_histories = load_all_histories(adv_dir)
        print(f"Loaded {len(adv_histories)} adversarial histories")
    
    # Load T5 model and tokenizer
    print(f"\nLoading T5 model and tokenizer: {args.model_name}")
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    
    # Load validation data
    print("\nLoading BoolQ validation dataset...")
    raw_val = load_dataset("boolq")["validation"]
    val_proc = raw_val.map(
        lambda ex: preprocess_boolq(ex, tokenizer),
        batched=False
    )
    val_proc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dl = DataLoader(val_proc, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate all models
    print("\n" + "=" * 80)
    print("Evaluating Hard (Projected) Prompts vs Soft Prompts")
    print("For each lambda: Best (non-adv) / Worst (adv) based on hard prompt accuracy")
    print("=" * 80)
    
    all_results = []
    
    # Process non-adversarial models
    if non_adv_histories:
        print("\nEvaluating all non-adversarial models...")
        for i, ((lam, lr, pl), history) in enumerate(sorted(non_adv_histories.items()), 1):
            model_path = os.path.join(non_adv_dir, f"model_lambda_{lam}_lr_{lr}_promptlen_{pl}.pt")
            
            if not os.path.exists(model_path):
                print(f"  [{i}/{len(non_adv_histories)}] Skipping: {model_path} not found")
                continue
            
            print(f"  [{i}/{len(non_adv_histories)}] Evaluating λ={lam}, lr={lr:.0e}, pl={pl}...")
            
            # Load model
            model = T5PEZPrompt(base_model, prompt_length=pl).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Evaluate with hard prompts
            hard_acc = evaluate_accuracy_hard_prompt(model, val_dl, tokenizer, device)
            
            # Get soft prompt accuracy from history (final epoch)
            soft_acc = history['val_acc'][-1] if 'val_acc' in history and len(history['val_acc']) > 0 else None
            
            all_results.append({
                'lambda': lam,
                'lr': lr,
                'prompt_length': pl,
                'adversarial': False,
                'hard_prompt_acc': hard_acc,
                'soft_prompt_acc': soft_acc,
                'difference': hard_acc - soft_acc if soft_acc is not None else None
            })
            
            soft_str = f"{soft_acc:.4f}" if soft_acc is not None else "N/A"
            print(f"    Hard={hard_acc:.4f}, Soft={soft_str}")
    
    # Process adversarial models
    if adv_histories:
        print("\nEvaluating all adversarial models...")
        for i, ((lam, lr, pl), history) in enumerate(sorted(adv_histories.items()), 1):
            model_path = os.path.join(adv_dir, f"model_lambda_{lam}_lr_{lr}_promptlen_{pl}.pt")
            
            if not os.path.exists(model_path):
                print(f"  [{i}/{len(adv_histories)}] Skipping: {model_path} not found")
                continue
            
            print(f"  [{i}/{len(adv_histories)}] Evaluating λ={lam}, lr={lr:.0e}, pl={pl}...")
            
            # Load model
            model = T5PEZPrompt(base_model, prompt_length=pl).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Evaluate with hard prompts
            hard_acc = evaluate_accuracy_hard_prompt(model, val_dl, tokenizer, device)
            
            # Get soft prompt accuracy from history (final epoch)
            soft_acc = history['val_acc'][-1] if 'val_acc' in history and len(history['val_acc']) > 0 else None
            
            all_results.append({
                'lambda': lam,
                'lr': lr,
                'prompt_length': pl,
                'adversarial': True,
                'hard_prompt_acc': hard_acc,
                'soft_prompt_acc': soft_acc,
                'difference': hard_acc - soft_acc if soft_acc is not None else None
            })
            
            soft_str = f"{soft_acc:.4f}" if soft_acc is not None else "N/A"
            print(f"    Hard={hard_acc:.4f}, Soft={soft_str}")
    
    # Select best for non-adversarial and worst for adversarial for each lambda
    results = []
    
    # Group by lambda and adversarial flag
    non_adv_results = [r for r in all_results if not r['adversarial']]
    adv_results = [r for r in all_results if r['adversarial']]
    
    # For non-adversarial: select best hard prompt accuracy for each lambda
    if non_adv_results:
        print("\n" + "=" * 80)
        print("Non-Adversarial: Selecting Best Hard Prompt Accuracy for Each Lambda")
        print("=" * 80)
        lambda_values = sorted(set(r['lambda'] for r in non_adv_results))
        for lam in lambda_values:
            lam_results = [r for r in non_adv_results if r['lambda'] == lam]
            best_result = max(lam_results, key=lambda x: x['hard_prompt_acc'])
            results.append(best_result)
            
            soft_str = f"{best_result['soft_prompt_acc']:.4f}" if best_result['soft_prompt_acc'] is not None else "N/A"
            diff_str = f"{best_result['difference']:.4f}" if best_result['difference'] is not None else "N/A"
            print(f"  λ={lam}: lr={best_result['lr']:.0e}, pl={best_result['prompt_length']}, "
                  f"Hard={best_result['hard_prompt_acc']:.4f}, Soft={soft_str}, Diff={diff_str}")
    
    # For adversarial: select worst hard prompt accuracy for each lambda
    if adv_results:
        print("\n" + "=" * 80)
        print("Adversarial: Selecting Worst Hard Prompt Accuracy for Each Lambda")
        print("=" * 80)
        lambda_values = sorted(set(r['lambda'] for r in adv_results))
        for lam in lambda_values:
            lam_results = [r for r in adv_results if r['lambda'] == lam]
            worst_result = min(lam_results, key=lambda x: x['hard_prompt_acc'])
            results.append(worst_result)
            
            soft_str = f"{worst_result['soft_prompt_acc']:.4f}" if worst_result['soft_prompt_acc'] is not None else "N/A"
            diff_str = f"{worst_result['difference']:.4f}" if worst_result['difference'] is not None else "N/A"
            print(f"  λ={lam}: lr={worst_result['lr']:.0e}, pl={worst_result['prompt_length']}, "
                  f"Hard={worst_result['hard_prompt_acc']:.4f}, Soft={soft_str}, Diff={diff_str}")
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        
        # Determine output path
        if args.output_csv is None:
            output_csv = os.path.join(args.results_dir, "hard_vs_soft_prompts.csv")
        else:
            output_csv = args.output_csv
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print("\n" + "=" * 80)
        print(f"Results saved to: {output_csv}")
        print("=" * 80)
        print(df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        print(f"Total models evaluated: {len(all_results)}")
        print(f"Selected models: {len(results)}")
        if df['difference'].notna().any():
            print(f"Mean difference (Hard - Soft): {df['difference'].mean():.4f}")
            print(f"Std difference: {df['difference'].std():.4f}")
            print(f"Min difference: {df['difference'].min():.4f}")
            print(f"Max difference: {df['difference'].max():.4f}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()

