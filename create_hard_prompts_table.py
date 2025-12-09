#!/usr/bin/env python3
"""
Create a formatted table from hard_vs_soft_prompts.csv showing
best/worst accuracy for each (lambda, prompt_length) pair.
"""

import pandas as pd
import sys
import os

def create_table(csv_path):
    """Load CSV and create formatted table."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    print("=" * 100)
    print("Hard Prompt Accuracy: Best (Non-Adversarial) / Worst (Adversarial) per (Lambda, Prompt Length)")
    print("=" * 100)
    
    # Separate adversarial and non-adversarial
    non_adv = df[df['adversarial'] == False].copy()
    adv = df[df['adversarial'] == True].copy()
    
    # Create tables
    if len(non_adv) > 0:
        print("\n" + "=" * 100)
        print("NON-ADVERSARIAL: Best Hard Prompt Accuracy per (Lambda, Prompt Length)")
        print("=" * 100)
        print(f"{'Lambda':<10} {'Prompt Length':<15} {'LR':<12} {'Hard Acc':<12} {'Soft Acc':<12} {'Difference':<12}")
        print("-" * 100)
        
        for _, row in non_adv.iterrows():
            print(f"{row['lambda']:<10.2f} {row['prompt_length']:<15} "
                  f"{row['lr']:<12.2e} {row['hard_prompt_acc']:<12.4f} "
                  f"{row['soft_prompt_acc']:<12.4f} {row['difference']:<12.4f}")
    
    if len(adv) > 0:
        print("\n" + "=" * 100)
        print("ADVERSARIAL: Worst Hard Prompt Accuracy per (Lambda, Prompt Length)")
        print("=" * 100)
        print(f"{'Lambda':<10} {'Prompt Length':<15} {'LR':<12} {'Hard Acc':<12} {'Soft Acc':<12} {'Difference':<12}")
        print("-" * 100)
        
        for _, row in adv.iterrows():
            print(f"{row['lambda']:<10.2f} {row['prompt_length']:<15} "
                  f"{row['lr']:<12.2e} {row['hard_prompt_acc']:<12.4f} "
                  f"{row['soft_prompt_acc']:<12.4f} {row['difference']:<12.4f}")
    
    # Create pivot tables for easier viewing
    print("\n" + "=" * 100)
    print("NON-ADVERSARIAL: Hard Prompt Accuracy Pivot Table")
    print("=" * 100)
    if len(non_adv) > 0:
        pivot_non_adv = non_adv.pivot_table(
            values='hard_prompt_acc',
            index='lambda',
            columns='prompt_length',
            aggfunc='first'
        )
        print(pivot_non_adv.to_string())
    
    print("\n" + "=" * 100)
    print("ADVERSARIAL: Hard Prompt Accuracy Pivot Table")
    print("=" * 100)
    if len(adv) > 0:
        pivot_adv = adv.pivot_table(
            values='hard_prompt_acc',
            index='lambda',
            columns='prompt_length',
            aggfunc='first'
        )
        print(pivot_adv.to_string())
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(f"Total rows: {len(df)}")
    print(f"Non-adversarial: {len(non_adv)}")
    print(f"Adversarial: {len(adv)}")
    
    if len(non_adv) > 0:
        print(f"\nNon-Adversarial Hard Acc - Mean: {non_adv['hard_prompt_acc'].mean():.4f}, "
              f"Std: {non_adv['hard_prompt_acc'].std():.4f}, "
              f"Min: {non_adv['hard_prompt_acc'].min():.4f}, "
              f"Max: {non_adv['hard_prompt_acc'].max():.4f}")
    
    if len(adv) > 0:
        print(f"Adversarial Hard Acc - Mean: {adv['hard_prompt_acc'].mean():.4f}, "
              f"Std: {adv['hard_prompt_acc'].std():.4f}, "
              f"Min: {adv['hard_prompt_acc'].min():.4f}, "
              f"Max: {adv['hard_prompt_acc'].max():.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "/mnt/polished-lake/home/annabelma/other/results_fixed/pez_fixed/hard_vs_soft_prompts.csv"
    
    create_table(csv_path)

