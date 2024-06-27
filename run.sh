#!/bin/bash

# Make sure the script exits if any command fails
set -e

# Define paths to your input and output files
VALID_INPUT="data/valid.json"
VALID_OUTPUT="data/valid.txt"
TRAIN_INPUT="data/train.json"
TRAIN_OUTPUT="data/train.txt"
TEST_INPUT="data/test.json"
TEST_OUTPUT="data/test.txt"

# Function to check if a file exists
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "File $1 does not exist. Exiting."
        exit 1
    fi
}

# Check if input files exist
check_file_exists $VALID_INPUT
check_file_exists $TRAIN_INPUT
check_file_exists $TEST_INPUT

# Run the extract_tags.py script on valid.json
python extract_tags.py $VALID_INPUT $VALID_OUTPUT

# Run the extract_tags.py script on train.json
python extract_tags.py $TRAIN_INPUT $TRAIN_OUTPUT

# Run the extract_tags.py script on test.json
python extract_tags.py $TEST_INPUT $TEST_OUTPUT

# Check if trainer.py exists
check_file_exists "trainer.py"

# Run the trainer.py script
python trainer.py

echo "All scripts executed successfully."
