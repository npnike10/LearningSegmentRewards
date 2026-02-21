"""Unit extraction for fine-grained credit assignment in RLHF."""

import re
from typing import List, Tuple
import torch


def extract_units_from_text_regex(text: str, max_units: int = 10) -> List[Tuple[int, int]]:
    """Extract step units from procedure text.

    Args:
        text: The procedure text to extract units from
        max_units: Maximum number of units to extract (remaining grouped as one)

    Returns:
        List of (start_char_idx, end_char_idx) tuples for each unit
    """
    # Pattern for step labels at line start
    # Matches: A., B., AA., AB., 1., 2., *1., *2., etc.
    label_pattern = r"^(\*?[A-Z]{1,2}\.)\s+"

    lines = [line for line in text.split("\n") if line]
    units = []
    step_positions = []  # Store positions where step labels start
    current_char_pos = 0

    # First pass: find all step label positions
    for i, line in enumerate(lines):
        line_start_pos = current_char_pos

        # Check if line starts with a valid step label
        match = re.match(label_pattern, line)

        if match:
            step_positions.append(line_start_pos)

            # If we've reached max_units, stop finding more steps
            if len(step_positions) >= max_units:
                break

        # Update character position (line length + newline)
        current_char_pos += len(line) + 1

    # Second pass: create units from step positions
    for i, step_start in enumerate(step_positions):
        if i < len(step_positions) - 1:
            # Unit ends just before next step starts (excluding newline)
            step_end = step_positions[i + 1] - 1
        else:
            # Last unit goes to end of text
            step_end = len(text)

        units.append((step_start, step_end))
    return units


def extract_units_from_text_entropy(text: str, entropy, max_units: int = 10) -> List[Tuple[int, int]]:
    entropy_threshold = 1.5
    # entropy[i] is the entropy when predicting token i+1
    # When entropy[i] > threshold, it means high uncertainty predicting token i+1
    # So token i+1 is where we want to split (start of next unit)
    boundary_token_indices = [0]
    for i, ent in enumerate(entropy):
        if ent > entropy_threshold and i < len(entropy) - 1:
            # High entropy at position i means uncertain about token i+1
            # So unit ends at i+1 (exclusive), next unit starts at i+1
            boundary_token_indices.append(i + 1)
    boundary_token_indices.append(len(entropy))

    # each pair in boundary tokens is a unit [start, end)
    token_units = []
    for i in range(0, len(boundary_token_indices) - 1):
        token_units.append((boundary_token_indices[i], boundary_token_indices[i + 1]))
    return token_units


def extract_units_from_tokens(
    token_ids: List[int], tokenizer, unit_extraction: str, max_units: int = 10, entropy=None
) -> List[Tuple[int, int]]:
    """Extract step units and return token-level indices.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer to decode tokens
        unit_extraction: method used to extract units from text. Options are 'regex', 'entropy', 'confidence'
        max_units: Maximum number of units to extract
        entropy: entropy values for output texts

    Returns:
        List of (start_token_idx, end_token_idx) tuples for each unit
    """
    # take all non-zero token ids (i.e., all action tokens) and skip the first four tokens (opening think token, followed by double newline token followed by ending think token followed by double newline token) and last one (im_end action token)
    mask = (token_ids != 0).cpu()
    output_text_token_ids = token_ids.cpu()[mask][4:-1].tolist()
    output_text_entropy = entropy.cpu()[mask][4:-1].tolist()
    # Decode to text
    text = tokenizer.decode(output_text_token_ids, skip_special_tokens=True)

    # index of first non-zero entry in token ids
    first_non_zero_token_id = token_ids.nonzero()[0].item()
    num_input_prompt_tokens = first_non_zero_token_id + 1
    num_think_part_tokens = 4

    # Get character-level units
    if unit_extraction == "regex":
        char_units = extract_units_from_text_regex(text, max_units)
        if not char_units:
            return []

        # Map character positions to token positions
        token_units = []
        char_to_token = build_char_to_token_map(output_text_token_ids, tokenizer)

        for char_start, char_end in char_units:
            # Find token indices corresponding to character positions
            token_start = char_to_token.get(char_start, 0)
            token_end = char_to_token.get(char_end - 1, len(output_text_token_ids) - 1) + 1

            # convert token indices from output token ids to sequence token ids
            token_start += num_input_prompt_tokens + num_think_part_tokens
            token_end += num_input_prompt_tokens + num_think_part_tokens

            token_units.append((token_start, token_end))
    else:
        token_units = extract_units_from_text_entropy(text, output_text_entropy, max_units)

        for i, unit in enumerate(token_units):
            token_units[i] = (
                unit[0] + num_input_prompt_tokens + num_think_part_tokens,
                unit[1] + num_input_prompt_tokens + num_think_part_tokens,
            )

    return token_units


def build_char_to_token_map(token_ids: List[int], tokenizer) -> dict:
    """Build mapping from character position to token index.

    Args:
        token_ids: List of token IDs
        tokenizer: Tokenizer

    Returns:
        Dictionary mapping character position to token index
    """
    char_to_token = {}
    char_pos = 0

    for token_idx, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        token_len = len(token_text)

        # Map all character positions in this token to the token index
        for i in range(token_len):
            char_to_token[char_pos + i] = token_idx

        char_pos += token_len

    return char_to_token


def get_unit_count(text: str) -> int:
    """Count number of step units in text.

    Args:
        text: The procedure text

    Returns:
        Number of units found
    """
    units = extract_units_from_text(text, max_units=1000)  # No limit for counting
    return len(units)
