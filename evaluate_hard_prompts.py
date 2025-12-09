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
def decode_prompt_text(model, tokenizer):
    """Decode the hard prompt tokens to text."""
    prompt_token_ids = model.get_prompt_token_ids()
    prompt_text = tokenizer.decode(
        prompt_token_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()
    return prompt_text


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


def find_all_model_files(directory):
    """Find all model files in a directory and parse their parameters."""
    model_files = []
    
    # Find all model files
    model_paths = glob.glob(os.path.join(directory, "model_*.pt"))
    
    for model_path in model_paths:
        filename = os.path.basename(model_path)
        # Try new format first (with promptlen)
        match = re.match(r"model_lambda_([0-9.]+)_lr_([0-9.e-]+)_promptlen_([0-9]+)\.pt", filename)
        if match:
            lambda_val = float(match.group(1))
            lr_val = float(match.group(2))
            prompt_length = int(match.group(3))
            
            model_files.append({
                'path': model_path,
                'lambda': lambda_val,
                'lr': lr_val,
                'prompt_length': prompt_length
            })
        else:
            # Try old format (without promptlen) for backward compatibility
            match_old = re.match(r"model_lambda_([0-9.]+)_lr_([0-9.e-]+)\.pt", filename)
            if match_old:
                lambda_val = float(match_old.group(1))
                lr_val = float(match_old.group(2))
                # Try to get prompt_length from history file
                history_path = os.path.join(directory, f"history_lambda_{lambda_val}_lr_{lr_val}.json")
                prompt_length = 10  # default
                if os.path.exists(history_path):
                    try:
                        with open(history_path, 'r') as f:
                            history = json.load(f)
                            prompt_length = history.get('prompt_length', 10)
                    except:
                        pass
                
                model_files.append({
                    'path': model_path,
                    'lambda': lambda_val,
                    'lr': lr_val,
                    'prompt_length': prompt_length
                })
    
    return model_files


def load_existing_csv(csv_path):
    """Load existing CSV and return set of already-evaluated models."""
    if not os.path.exists(csv_path):
        return set(), pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    # Create set of (lambda, lr, prompt_length, adversarial) tuples
    evaluated = set()
    for _, row in df.iterrows():
        key = (row['lambda'], row['lr'], row['prompt_length'], bool(row['adversarial']))
        evaluated.add(key)
    
    return evaluated, df


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
    
    # Determine output path
    if args.output_csv is None:
        output_csv = os.path.join(args.results_dir, "hard_vs_soft_prompts.csv")
    else:
        output_csv = args.output_csv
    
    # Load existing CSV to check what's already evaluated
    print("\nChecking existing results...")
    evaluated_set, existing_df = load_existing_csv(output_csv)
    if len(evaluated_set) > 0:
        print(f"Found {len(evaluated_set)} already-evaluated models in {output_csv}")
    else:
        print("No existing results found. Will evaluate all models.")
    
    # Load histories (for soft prompt accuracy comparison)
    print("\nLoading histories...")
    non_adv_histories = {}
    if os.path.exists(non_adv_dir):
        non_adv_histories = load_all_histories(non_adv_dir)
        print(f"Loaded {len(non_adv_histories)} non-adversarial histories")
    
    adv_histories = {}
    if os.path.exists(adv_dir):
        adv_histories = load_all_histories(adv_dir)
        print(f"Loaded {len(adv_histories)} adversarial histories")
    
    # Find all model files
    print("\nFinding all model files...")
    non_adv_models = []
    if os.path.exists(non_adv_dir):
        non_adv_models = find_all_model_files(non_adv_dir)
        print(f"Found {len(non_adv_models)} non-adversarial model files")
    
    adv_models = []
    if os.path.exists(adv_dir):
        adv_models = find_all_model_files(adv_dir)
        print(f"Found {len(adv_models)} adversarial model files")
    
    # Filter out already-evaluated models
    non_adv_to_eval = []
    for model_info in non_adv_models:
        key = (model_info['lambda'], model_info['lr'], model_info['prompt_length'], False)
        if key not in evaluated_set:
            non_adv_to_eval.append(model_info)
    
    adv_to_eval = []
    for model_info in adv_models:
        key = (model_info['lambda'], model_info['lr'], model_info['prompt_length'], True)
        if key not in evaluated_set:
            adv_to_eval.append(model_info)
    
    total_to_eval = len(non_adv_to_eval) + len(adv_to_eval)
    print(f"\nModels to evaluate: {total_to_eval} ({len(non_adv_to_eval)} non-adversarial, {len(adv_to_eval)} adversarial)")
    
    # Initialize all_results with existing results
    if len(existing_df) > 0:
        all_results = existing_df.to_dict('records')
        # Ensure prompt_text column exists for existing results (set to None if missing)
        for result in all_results:
            if 'prompt_text' not in result:
                result['prompt_text'] = None
    else:
        all_results = []
    
    if total_to_eval == 0:
        print("All models already evaluated! Re-running selection logic on existing results.")
    else:
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
        
        # Evaluate new models
        print("\n" + "=" * 80)
        print("Evaluating Hard (Projected) Prompts vs Soft Prompts")
        print("=" * 80)
        
        new_results = []
        
        # Process non-adversarial models
        if non_adv_to_eval:
            print(f"\nEvaluating {len(non_adv_to_eval)} non-adversarial models...")
            for i, model_info in enumerate(sorted(non_adv_to_eval, key=lambda x: (x['lambda'], x['lr'], x['prompt_length'])), 1):
                lam = model_info['lambda']
                lr = model_info['lr']
                pl = model_info['prompt_length']
                model_path = model_info['path']
                
                print(f"  [{i}/{len(non_adv_to_eval)}] Evaluating λ={lam}, lr={lr:.0e}, pl={pl}...")
                
                # Load model
                model = T5PEZPrompt(base_model, prompt_length=pl).to(device)
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                
                # Evaluate with hard prompts
                hard_acc = evaluate_accuracy_hard_prompt(model, val_dl, tokenizer, device)
                
                # Decode prompt text
                prompt_text = decode_prompt_text(model, tokenizer)
                
                # Get soft prompt accuracy from history (final epoch)
                history_key = (lam, lr, pl)
                soft_acc = None
                if history_key in non_adv_histories:
                    history = non_adv_histories[history_key]
                    soft_acc = history['val_acc'][-1] if 'val_acc' in history and len(history['val_acc']) > 0 else None
                
                new_results.append({
                    'lambda': lam,
                    'lr': lr,
                    'prompt_length': pl,
                    'adversarial': False,
                    'hard_prompt_acc': hard_acc,
                    'soft_prompt_acc': soft_acc,
                    'difference': hard_acc - soft_acc if soft_acc is not None else None,
                    'prompt_text': prompt_text
                })
                
                soft_str = f"{soft_acc:.4f}" if soft_acc is not None else "N/A"
                print(f"    Hard={hard_acc:.4f}, Soft={soft_str}, Prompt: {prompt_text[:50]}..." if len(prompt_text) > 50 else f"    Hard={hard_acc:.4f}, Soft={soft_str}, Prompt: {prompt_text}")
        
        # Process adversarial models
        if adv_to_eval:
            print(f"\nEvaluating {len(adv_to_eval)} adversarial models...")
            for i, model_info in enumerate(sorted(adv_to_eval, key=lambda x: (x['lambda'], x['lr'], x['prompt_length'])), 1):
                lam = model_info['lambda']
                lr = model_info['lr']
                pl = model_info['prompt_length']
                model_path = model_info['path']
                
                print(f"  [{i}/{len(adv_to_eval)}] Evaluating λ={lam}, lr={lr:.0e}, pl={pl}...")
                
                # Load model
                model = T5PEZPrompt(base_model, prompt_length=pl).to(device)
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                
                # Evaluate with hard prompts
                hard_acc = evaluate_accuracy_hard_prompt(model, val_dl, tokenizer, device)
                
                # Decode prompt text
                prompt_text = decode_prompt_text(model, tokenizer)
                
                # Get soft prompt accuracy from history (final epoch)
                history_key = (lam, lr, pl)
                soft_acc = None
                if history_key in adv_histories:
                    history = adv_histories[history_key]
                    soft_acc = history['val_acc'][-1] if 'val_acc' in history and len(history['val_acc']) > 0 else None
                
                new_results.append({
                    'lambda': lam,
                    'lr': lr,
                    'prompt_length': pl,
                    'adversarial': True,
                    'hard_prompt_acc': hard_acc,
                    'soft_prompt_acc': soft_acc,
                    'difference': hard_acc - soft_acc if soft_acc is not None else None,
                    'prompt_text': prompt_text
                })
                
                soft_str = f"{soft_acc:.4f}" if soft_acc is not None else "N/A"
                print(f"    Hard={hard_acc:.4f}, Soft={soft_str}, Prompt: {prompt_text[:50]}..." if len(prompt_text) > 50 else f"    Hard={hard_acc:.4f}, Soft={soft_str}, Prompt: {prompt_text}")
        
        # Combine new results with existing results
        all_results = all_results + new_results
    
    # Select best for non-adversarial and worst for adversarial for each (lambda, prompt_length) pair
    print("\n" + "=" * 80)
    print("Selecting Best (non-adv) / Worst (adv) Hard Prompt Accuracy for Each (Lambda, Prompt Length) Pair")
    print("=" * 80)
    
    results = []
    
    # Group by lambda and adversarial flag
    non_adv_results = [r for r in all_results if not r['adversarial']]
    adv_results = [r for r in all_results if r['adversarial']]
    
    # For non-adversarial: select best hard prompt accuracy for each (lambda, prompt_length) pair
    if non_adv_results:
        print("\nNon-Adversarial: Selecting Best Hard Prompt Accuracy for Each (Lambda, Prompt Length) Pair")
        # Get all unique (lambda, prompt_length) pairs
        lambda_pl_pairs = sorted(set((r['lambda'], r['prompt_length']) for r in non_adv_results))
        for lam, pl in lambda_pl_pairs:
            lam_pl_results = [r for r in non_adv_results if r['lambda'] == lam and r['prompt_length'] == pl]
            best_result = max(lam_pl_results, key=lambda x: x['hard_prompt_acc'])
            results.append(best_result)
            
            soft_str = f"{best_result['soft_prompt_acc']:.4f}" if best_result['soft_prompt_acc'] is not None else "N/A"
            diff_str = f"{best_result['difference']:.4f}" if best_result['difference'] is not None else "N/A"
            prompt_str = best_result.get('prompt_text', 'N/A')
            prompt_display = prompt_str[:60] + "..." if prompt_str and len(prompt_str) > 60 else (prompt_str or "N/A")
            print(f"  λ={lam}, pl={pl}: lr={best_result['lr']:.0e}, "
                  f"Hard={best_result['hard_prompt_acc']:.4f}, Soft={soft_str}, Diff={diff_str}")
            print(f"    Prompt: {prompt_display}")
    
    # For adversarial: select worst hard prompt accuracy for each (lambda, prompt_length) pair
    if adv_results:
        print("\nAdversarial: Selecting Worst Hard Prompt Accuracy for Each (Lambda, Prompt Length) Pair")
        # Get all unique (lambda, prompt_length) pairs
        lambda_pl_pairs = sorted(set((r['lambda'], r['prompt_length']) for r in adv_results))
        for lam, pl in lambda_pl_pairs:
            lam_pl_results = [r for r in adv_results if r['lambda'] == lam and r['prompt_length'] == pl]
            worst_result = min(lam_pl_results, key=lambda x: x['hard_prompt_acc'])
            results.append(worst_result)
            
            soft_str = f"{worst_result['soft_prompt_acc']:.4f}" if worst_result['soft_prompt_acc'] is not None else "N/A"
            diff_str = f"{worst_result['difference']:.4f}" if worst_result['difference'] is not None else "N/A"
            prompt_str = worst_result.get('prompt_text', 'N/A')
            prompt_display = prompt_str[:60] + "..." if prompt_str and len(prompt_str) > 60 else (prompt_str or "N/A")
            print(f"  λ={lam}, pl={pl}: lr={worst_result['lr']:.0e}, "
                  f"Hard={worst_result['hard_prompt_acc']:.4f}, Soft={soft_str}, Diff={diff_str}")
            print(f"    Prompt: {prompt_display}")
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        
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

