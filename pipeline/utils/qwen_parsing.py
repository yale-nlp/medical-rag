import re

def extract_qwen_response(raw_output: str) -> str:
    """
    Parses the raw output from a Qwen model that uses a <think> block.
    It removes the entire <think>...</think> block and returns the remaining text.

    Args:
        raw_output (str): The full string generated by the model.

    Returns:
        str: The cleaned response with the thinking block removed.
    """
    # Use a non-greedy regular expression to find the think block.
    # re.DOTALL allows '.' to match newline characters.
    match = re.search(r"<think>.*?</think>", raw_output, re.DOTALL)
    
    if match:
        # If a think block is found, remove it and strip leading/trailing whitespace
        # from the remaining string.
        return raw_output.replace(match.group(0), "").strip()
        
    # If no think block is found, return the original output as is.
    return raw_output