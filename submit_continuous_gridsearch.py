#!/usr/bin/env python3
"""
Submit Continuous Soft Prompt Grid Search Array Job
Reads a YAML config file and submits an array job for grid search.
"""

import yaml
import os
import subprocess
import itertools
import sys


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_job_list(config):
    """Create list of all job parameters from grid."""
    # Get base LR grid (fallback if specific grids not provided)
    base_lr_grid = config.get('lr_grid', [])
    
    # Get specific LR grids for adversarial/non-adversarial if provided
    lr_grid_non_adv = config.get('lr_grid_non_adv', base_lr_grid)
    lr_grid_adv = config.get('lr_grid_adv', base_lr_grid)
    
    prompt_length_grid = config.get('prompt_length_grid', [10])  # Default: [10]
    adversarial_flags = config.get('adversarial', [False, True])  # Default: both
    num_epochs = config.get('num_epochs', 5)  # Default: 5 epochs
    
    jobs = []
    for adv in adversarial_flags:
        # Select appropriate LR grid based on adversarial flag
        lr_grid = lr_grid_adv if adv else lr_grid_non_adv
        
        for lr, prompt_len in itertools.product(lr_grid, prompt_length_grid):
            jobs.append({
                'lr': lr,
                'prompt_length': prompt_len,
                'adversarial': adv,
                'num_epochs': num_epochs
            })
    
    return jobs


def write_job_list_file(jobs, output_path):
    """Write job list to file (one job per line)."""
    with open(output_path, 'w') as f:
        for job in jobs:
            adv_flag = '--adversarial' if job['adversarial'] else ''
            line = f"{job['lr']} {job['prompt_length']} {job['num_epochs']} {adv_flag}\n"
            f.write(line)
    return output_path


def submit_array_job(config, jobs, script_dir):
    """Submit SLURM array job."""
    num_jobs = len(jobs)
    
    # Create log directory
    log_dir = os.path.join(config['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Write job list file
    job_list_file = os.path.join(script_dir, 'continuous_job_list.txt')
    write_job_list_file(jobs, job_list_file)
    
    # Read sbatch template
    sbatch_template = os.path.join(script_dir, 'run_continuous_gridsearch.sbatch')
    with open(sbatch_template, 'r') as f:
        sbatch_content = f.read()
    
    # Replace placeholders
    sbatch_content = sbatch_content.replace('{{NUM_JOBS}}', str(num_jobs))
    sbatch_content = sbatch_content.replace('{{OUTPUT_DIR}}', config['output_dir'])
    sbatch_content = sbatch_content.replace('{{JOB_LIST_FILE}}', job_list_file)
    sbatch_content = sbatch_content.replace('{{LOG_DIR}}', log_dir)
    
    # Write customized sbatch script
    temp_sbatch = os.path.join(script_dir, 'run_continuous_gridsearch_temp.sbatch')
    with open(temp_sbatch, 'w') as f:
        f.write(sbatch_content)
    
    # Submit job
    print(f"Submitting array job with {num_jobs} tasks...")
    result = subprocess.run(
        ['sbatch', temp_sbatch],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Job submitted successfully! Job ID: {job_id}")
        print(f"Monitor with: squeue -j {job_id}")
        return job_id
    else:
        print(f"Error submitting job: {result.stderr}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python submit_continuous_gridsearch.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    config = load_config(config_path)
    
    # Validate config
    required_keys = ['output_dir']
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key in config: {key}")
            sys.exit(1)
    
    # Check that at least one LR grid is provided
    if 'lr_grid' not in config and 'lr_grid_non_adv' not in config and 'lr_grid_adv' not in config:
        print("Error: Must provide at least one of: lr_grid, lr_grid_non_adv, or lr_grid_adv")
        sys.exit(1)
    
    # Create job list
    jobs = create_job_list(config)
    
    # Get LR grids for display
    base_lr_grid = config.get('lr_grid', [])
    lr_grid_non_adv = config.get('lr_grid_non_adv', base_lr_grid)
    lr_grid_adv = config.get('lr_grid_adv', base_lr_grid)
    
    print(f"Created {len(jobs)} jobs from grid:")
    if 'lr_grid_non_adv' in config or 'lr_grid_adv' in config:
        print(f"  Learning rates (non-adversarial): {lr_grid_non_adv}")
        print(f"  Learning rates (adversarial): {lr_grid_adv}")
    else:
        print(f"  Learning rates: {base_lr_grid}")
    print(f"  Prompt lengths: {config.get('prompt_length_grid', [10])}")
    print(f"  Adversarial: {config.get('adversarial', [False, True])}")
    print(f"  Epochs: {config.get('num_epochs', 5)}")
    
    # Get script directory (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Submit array job
    job_id = submit_array_job(config, jobs, script_dir)
    
    if job_id:
        print(f"\nJob list saved to: {os.path.join(script_dir, 'continuous_job_list.txt')}")
        print(f"Output directory: {config['output_dir']}")


if __name__ == "__main__":
    main()

