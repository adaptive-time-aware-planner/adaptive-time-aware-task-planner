#!/usr/bin/env python3

import os
import glob
import subprocess
import argparse
from pathlib import Path

def find_bddl_files(base_dir: str) -> list:
    """Find all BDDL files in activity_definitions directory and its subdirectories.
    
    Args:
        base_dir (str): Base directory containing activity_definitions
        
    Returns:
        list: List of paths to BDDL files
    """
    bddl_pattern = os.path.join(base_dir, "data", "activity_definitions", "**", "problem*.bddl")
    return sorted(glob.glob(bddl_pattern, recursive=True))

def process_bddl_file(bddl_file: str, gpt_version: str, api_key_file: str) -> None:
    """Process a single BDDL file using pddlrun_llmseparate.py.
    
    Args:
        bddl_file (str): Path to BDDL file
        gpt_version (str): GPT version to use
        api_key_file (str): Path to API key file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        "python",
        os.path.join(script_dir, "pddlrun_llmseparate.py"),
        "--bddl-file", os.path.abspath(bddl_file),
        "--gpt-version", gpt_version
    ]
    
    if api_key_file:
        cmd.extend(["--openai-api-key-file", api_key_file])
    
    print(f"\n{'='*50}")
    print(f"Processing: {bddl_file}")
    print(f"{'='*50}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {bddl_file}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error processing {bddl_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process all BDDL files in activity_definitions directory")
    parser.add_argument("--gpt-version", type=str, default="gpt-4o",
                      choices=['gpt-3.5-turbo', 'gpt-4o', 'gpt-3.5-turbo-16k'],
                      help="GPT version to use")
    parser.add_argument("--openai-api-key-file", type=str, default="api_key",
                      help="Path to OpenAI API key file")
    parser.add_argument("--activity", type=str, default=None,
                      help="Process specific activity directory only")
    
    args = parser.parse_args()
    
    # Get base directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    # Find all BDDL files
    if args.activity:
        # If specific activity specified, only process those files
        bddl_pattern = os.path.join(base_dir, "data", "activity_definitions", 
                                  args.activity, "problem*.bddl")
        bddl_files = sorted(glob.glob(bddl_pattern))
    else:
        bddl_files = find_bddl_files(base_dir)
    
    if not bddl_files:
        print("No BDDL files found!")
        return
    
    print(f"Found {len(bddl_files)} BDDL files to process")
    
    # Process each BDDL file
    for i, bddl_file in enumerate(bddl_files, 1):
        print(f"\nProcessing file {i}/{len(bddl_files)}")
        process_bddl_file(bddl_file, args.gpt_version, args.openai_api_key_file)
    
    print("\nAll BDDL files processed!")

if __name__ == "__main__":
    main() 