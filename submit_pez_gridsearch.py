#!/usr/bin/env python3
"""
Submit PEZ Grid Search Array Job
Reads a YAML config file and submits an array job for grid search.
"""

import yaml
import os
import subprocess
import itertools
import sys
import glob


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def check_result_exists(output_dir, lambda_ppl, lr, prompt_length, adversarial):
    """Check if the history JSON file already exists for this job configuration."""
    # Determine subdirectory based on adversarial flag
    adv_subdir = "adversarial_true" if adversarial else "adversarial_false"
    subdir = os.path.join(output_dir, adv_subdir)
    
    if not os.path.exists(subdir):
        return False
    
    # Use glob to find files matching the pattern, since LR might be formatted differently
    # Pattern: history_lambda_{lambda}_lr_*_promptlen_{prompt_length}.json
    pattern = os.path.join(subdir, f"history_lambda_{lambda_ppl}_lr_*_promptlen_{prompt_length}.json")
    matching_files = glob.glob(pattern)
    
    return len(matching_files) > 0


def create_job_list(config):
    """Create list of all job parameters from grid, skipping jobs that already have results."""
    lambda_grid = config['lambda_grid']
    output_dir = config['output_dir']
    jobs = []
    skipped = []
    
    # Non-adversarial jobs
    if 'non_adversarial' in config:
        non_adv_config = config['non_adversarial']
        lr_grid = non_adv_config['lr_grid']
        num_epochs = non_adv_config['num_epochs']
        prompt_lengths = non_adv_config.get('prompt_lengths', [10])  # Default: [10]
        
        for lam, lr, pl in itertools.product(lambda_grid, lr_grid, prompt_lengths):
            # Check if result already exists
            if check_result_exists(output_dir, lam, lr, pl, False):
                skipped.append(f"lambda={lam}, lr={lr}, prompt_length={pl}, adversarial=False")
                continue
            
            jobs.append({
                'lambda_ppl': lam,
                'lr': lr,
                'adversarial': False,
                'num_epochs': num_epochs,
                'prompt_length': pl
            })
    
    # Adversarial jobs
    if 'adversarial' in config:
        adv_config = config['adversarial']
        if adv_config.get('enabled', True):  # Only add if enabled
            lr_grid = adv_config['lr_grid']
            num_epochs = adv_config['num_epochs']
            prompt_lengths = adv_config.get('prompt_lengths', [10])  # Default: [10]
            
            for lam, lr, pl in itertools.product(lambda_grid, lr_grid, prompt_lengths):
                # Check if result already exists
                if check_result_exists(output_dir, lam, lr, pl, True):
                    skipped.append(f"lambda={lam}, lr={lr}, prompt_length={pl}, adversarial=True")
                    continue
                
                jobs.append({
                    'lambda_ppl': lam,
                    'lr': lr,
                    'adversarial': True,
                    'num_epochs': num_epochs,
                    'prompt_length': pl
                })
    
    # Print summary of skipped jobs
    if skipped:
        print(f"\nSkipped {len(skipped)} jobs that already have results:")
        for skip_info in skipped[:10]:  # Show first 10
            print(f"  - {skip_info}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    
    return jobs


def write_job_list_file(jobs, output_path):
    """Write job list to file (one job per line)."""
    with open(output_path, 'w') as f:
        for job in jobs:
            adv_flag = '--adversarial' if job['adversarial'] else ''
            prompt_length = job.get('prompt_length', 10)  # Default: 10
            line = f"{job['lambda_ppl']} {job['lr']} {job['num_epochs']} {prompt_length} {adv_flag}\n"
            f.write(line)
    return output_path


def submit_array_job(config, jobs, script_dir):
    """Submit SLURM array job."""
    num_jobs = len(jobs)
    
    # Create log directory
    log_dir = os.path.join(config['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Write job list file
    job_list_file = os.path.join(script_dir, 'pez_job_list.txt')
    write_job_list_file(jobs, job_list_file)
    
    # Read sbatch template
    sbatch_template = os.path.join(script_dir, 'run_pez_gridsearch.sbatch')
    with open(sbatch_template, 'r') as f:
        sbatch_content = f.read()
    
    # Replace placeholders
    sbatch_content = sbatch_content.replace('{{NUM_JOBS}}', str(num_jobs))
    sbatch_content = sbatch_content.replace('{{OUTPUT_DIR}}', config['output_dir'])
    sbatch_content = sbatch_content.replace('{{JOB_LIST_FILE}}', job_list_file)
    sbatch_content = sbatch_content.replace('{{LOG_DIR}}', log_dir)
    
    # Write customized sbatch script
    temp_sbatch = os.path.join(script_dir, 'run_pez_gridsearch_temp.sbatch')
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
        print("Usage: python submit_pez_gridsearch.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    config = load_config(config_path)
    
    # Validate config
    required_keys = ['lambda_grid', 'output_dir']
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key in config: {key}")
            sys.exit(1)
    
    # Validate that at least one training mode is configured
    if 'non_adversarial' not in config and 'adversarial' not in config:
        print("Error: At least one of 'non_adversarial' or 'adversarial' must be configured")
        sys.exit(1)
    
    if 'non_adversarial' in config:
        non_adv_required = ['lr_grid', 'num_epochs', 'prompt_lengths']
        for key in non_adv_required:
            if key not in config['non_adversarial']:
                print(f"Error: Missing required key in non_adversarial config: {key}")
                sys.exit(1)
    
    if 'adversarial' in config and config['adversarial'].get('enabled', True):
        adv_required = ['lr_grid', 'num_epochs', 'prompt_lengths']
        for key in adv_required:
            if key not in config['adversarial']:
                print(f"Error: Missing required key in adversarial config: {key}")
                sys.exit(1)
    
    # Create job list
    jobs = create_job_list(config)
    print(f"Created {len(jobs)} jobs from grid:")
    print(f"  Lambda values: {config['lambda_grid']}")
    
    if 'non_adversarial' in config:
        non_adv = config['non_adversarial']
        print(f"  Non-adversarial:")
        print(f"    Learning rates: {non_adv['lr_grid']}")
        print(f"    Epochs: {non_adv['num_epochs']}")
        print(f"    Prompt lengths: {non_adv.get('prompt_lengths', [10])}")
    
    if 'adversarial' in config and config['adversarial'].get('enabled', True):
        adv = config['adversarial']
        print(f"  Adversarial:")
        print(f"    Learning rates: {adv['lr_grid']}")
        print(f"    Epochs: {adv['num_epochs']}")
        print(f"    Prompt lengths: {adv.get('prompt_lengths', [10])}")
    
    # Get script directory (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Submit array job
    job_id = submit_array_job(config, jobs, script_dir)
    
    if job_id:
        print(f"\nJob list saved to: {os.path.join(script_dir, 'pez_job_list.txt')}")
        print(f"Output directory: {config['output_dir']}")


if __name__ == "__main__":
    main()

