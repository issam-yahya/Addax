"""
Run the extract_tags.py and trainer.py scripts.

Usage:
    python run.py
"""

import os
import subprocess
import sys

# Define paths to your input and output files
VALID_INPUT = "data/valid.json"
VALID_OUTPUT = "data/valid.txt"
TRAIN_INPUT = "data/train.json"
TRAIN_OUTPUT = "data/train.txt"
TEST_INPUT = "data/test.json"
TEST_OUTPUT = "data/test.txt"


def check_file_exists(file_path):
    """Check if the given file exists."""
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist. Exiting.")
        sys.exit(1)

def run_command(command):
    """Run a shell command."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command '{command}' failed with return code {result.returncode}. Exiting.")
        sys.exit(1)

def extract_if_needed(input_file, output_file):
    """Run extract_tags.py if the output file does not exist."""
    if os.path.isfile(output_file):
        print(f"{output_file} already exists. Skipping extraction.")
    else:
        run_command(f"python extract_tags.py {input_file} {output_file}")

if __name__ == "__main__":
    # Check if input files exist
    check_file_exists(VALID_INPUT)
    check_file_exists(TRAIN_INPUT)
    check_file_exists(TEST_INPUT)
    check_file_exists("extract_tags.py")
    check_file_exists("trainer.py")

    # Run the extract_tags.py script on valid.json if needed
    extract_if_needed(VALID_INPUT, VALID_OUTPUT)

    # Run the extract_tags.py script on train.json if needed
    extract_if_needed(TRAIN_INPUT, TRAIN_OUTPUT)

    # Run the extract_tags.py script on test.json if needed
    extract_if_needed(TEST_INPUT, TEST_OUTPUT)

    # Run the trainer.py script
    run_command("python trainer.py")

    # Find the best checkpoint
    run_command("python find_best_checkpoint.py .")

    print("All scripts executed successfully.")

