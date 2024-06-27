import json

def get_tags(tag_item):
    """
    Function to recursively extract tag values including those from nested tags
    :param tag_item: dict - The tag item to extract values from
    :return: list - List of extracted tag values
    """
    # Function to recursively extract tag values including those from nested tags
    tag_values = []
    if 'value' in tag_item:
        tag_values.append(tag_item['value'])
    if 'tags' in tag_item and tag_item['tags'] is not None:
        for subtag in tag_item['tags']:
            tag_values.extend(get_tags(subtag))
    return tag_values

def extract_tokens_tags(json_file_path, output_file_path):
    """
    Extracts tokens and their corresponding tags from a JSON file and writes the results to an output text file.
    
    Args:
        json_file_path (str): The path to the JSON file containing the data.
        output_file_path (str): The path to the output text file where the results will be written.
        
    Returns:
        None
        
    """
    # Read the JSON data from file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    # Initialize a list to hold token-tag pairs
    token_tag_pairs = []

    # Process each sentence in the JSON data
    for sentence in json_data:
        for token_data in sentence['tokens']:
            # Extract the token text
            token = token_data['token']
            # Extract all tag values into a flattened list (including nested tags)
            all_tags_list = [get_tags(tag) for tag in token_data['tags']]
            # Flatten the list of lists into a single list and then join with space
            all_tags_flat_list = [item for sublist in all_tags_list for item in sublist]
            all_tags = '+'.join(all_tags_flat_list)
            # Append the token and concatenated tags as a single string to the list
            token_tag_pairs.append(f"{token} {all_tags}")
        token_tag_pairs.append('')
    
    # Write the results to the output text file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for pair in token_tag_pairs:
            file.write(pair + '\n')

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract token-tag pairs from a JSON file and write to an output file.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path to the output text file.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the extraction function with provided file paths
    extract_tokens_tags(args.input_file, args.output_file)
