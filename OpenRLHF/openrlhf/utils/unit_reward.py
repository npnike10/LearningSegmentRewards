"""Unit-level difference reward computation for fine-grained credit assignment.
Note that unit extraction is designed and test for Qwen3-4B model, may need
changes for a different model, depending on that model's tokenization details."""

import torch
from typing import List, Tuple
import ray

from openrlhf.utils.unit_extraction import extract_units_from_tokens
from typing import Callable
from typing import Optional


def create_masked_sequence(
    sequence: torch.Tensor, unit_start: int, unit_end: int, tokenizer, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Create a sequence with the specified unit replaced with space tokens.

    Args:
        sequence: Original token sequence (seq_len,)
        unit_start: Start token index of unit
        unit_end: End token index of unit
        tokenizer: Tokenizer

    Returns:
        Modified sequence with unit tokens replaced by space tokens
    """
    # Get space token ID
    space_token_id = tokenizer.encode(" ", add_special_tokens=False)[0]

    # Clone sequence and replace unit tokens with space tokens
    modified_seq = sequence.clone()
    modified_seq[unit_start:unit_end] = space_token_id

    # clone attention mask
    attn_mask = attention_mask.clone()
    # replace tokens after current unit with im end + pad tokens
    modified_seq[unit_end] = tokenizer.eos_token_id
    if unit_end + 1 < len(modified_seq):
        modified_seq[unit_end + 1 :] = tokenizer.pad_token_id
        # modify attention mask to match with modified sequence padding
        attn_mask[unit_end + 1 :] = 0

    return modified_seq, attn_mask


def get_rewards_from_model(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    reward_model_group,
) -> torch.Tensor:
    """Get rewards from reward model for a batch of sequences.

    Args:
        sequences: Token sequences (batch_size, seq_len)
        attention_mask: Attention mask (batch_size, seq_len)
        reward_model_group: Ray actor group for reward model

    Returns:
        Rewards (batch_size,)
    """
    # Split sequences for distributed processing
    sequences_list = [sequences]
    attention_mask_list = [attention_mask]

    # Call reward model
    r_refs = reward_model_group.async_run_method_batch(
        method_name="forward",
        sequences=sequences_list,
        attention_mask=attention_mask_list,
        pad_sequence=[True] * len(sequences_list),
    )

    # Get results (handle ring_attn_size and ds_tensor_parallel_size duplication)
    results = ray.get(r_refs)

    # Extract rewards from first non-duplicate result
    rewards = results[0][0] if isinstance(results[0], list) else results[0]

    return rewards


def compute_unit_difference_rewards_batched(
    samples_list: List,
    reward_model_group,
    tokenizer,
    drca_beta: float,
    unit_extraction: str,
    max_units: int = 10,
    colocate_all_models: bool = False,
    duplicate_factor: int = 1,
) -> Tuple[List, bool]:
    """Compute unit-level difference rewards for a list of Experience samples.

    This function modifies samples_list in-place, replacing outcome-level rewards
    with per-token unit-level difference rewards.

    Args:
        samples_list: List of Experience objects with sequences and outcome rewards
        reward_model_group: Ray actor group for reward model
        tokenizer: Tokenizer
        drca_beta: DRCA reward shaping parameter
        max_units: Maximum number of units per sequence

    Returns:
        Modified samples_list with unit-level rewards
        Whether no units present in every sequence
    """
    sequences_list = [s.sequences for s in samples_list]
    attention_mask_list = [s.attention_mask for s in samples_list]
    action_mask_list = [s.action_mask for s in samples_list]
    outcome_rewards_list = [s.rewards for s in samples_list]
    entropy_list = [s.entropy for s in samples_list]

    # Initialize token rewards - zeros for DRCA, outcome reward only at EOS for PPO-style
    token_rewards_list = []
    for i, samples in enumerate(samples_list):
        batch_size, action_len = samples.action_mask.shape
        if drca_beta == 0:
            # PPO-style: only EOS gets outcome reward
            token_rewards = torch.zeros(
                batch_size, action_len, dtype=outcome_rewards_list[i].dtype, device=outcome_rewards_list[i].device
            )
        else:
            # DRCA-style: initialize with zeros (will be filled by unit rewards)
            token_rewards = torch.zeros(
                batch_size, action_len, dtype=outcome_rewards_list[i].dtype, device=outcome_rewards_list[i].device
            )
        token_rewards_list.append(token_rewards)

    # Process each sequence
    masked_sequences_list = []
    attention_mask_for_masked_sequences_list = (
        []
    )  # modified attention mask to incorporate modified padding in masked sequence
    sequence_to_units_list = []  # Track which units belong to which sequence

    for i, sequences in enumerate(sequences_list):
        masked_sequences = []
        attention_mask_for_masked_sequences = []
        sequence_to_units = []
        for j, seq in enumerate(sequences):
            attn_mask = attention_mask_list[i][j]
            act_mask = action_mask_list[i][j]

            # Get valid token IDs (only generated tokens)
            # Extract units from this sequence
            token_ids = seq[1:] * act_mask
            units = extract_units_from_tokens(
                token_ids, tokenizer, unit_extraction, max_units, entropy=entropy_list[i][j]
            )

            if not units:
                masked_sequences.append([])
                attention_mask_for_masked_sequences.append([])
                sequence_to_units.append([])
                continue

            # Create modified sequences for each unit (with unit removed/masked)
            masked_seqs = []
            attn_mask_for_masked_seqs = []
            seq_units = []
            for unit_start, unit_end in units:
                masked_seq, attn_mask_for_masked_seq = create_masked_sequence(
                    seq, unit_start, unit_end, tokenizer, attn_mask
                )
                masked_seqs.append(masked_seq)
                attn_mask_for_masked_seqs.append(attn_mask_for_masked_seq)
                seq_units.append((unit_start, unit_end))

            masked_sequences.append(torch.stack(masked_seqs))
            attention_mask_for_masked_sequences.append(torch.stack(attn_mask_for_masked_seqs))
            sequence_to_units.append(seq_units)
        masked_sequences_list.append(masked_sequences)
        attention_mask_for_masked_sequences_list.append(attention_mask_for_masked_sequences)
        sequence_to_units_list.append(sequence_to_units)

    # If no units found in any sequence, return PPO-style rewards (scalar outcome rewards)
    if not any(sequence_to_units_list):
        for i, samples in enumerate(samples_list):
            # Return scalar rewards for PPO compatibility
            samples.rewards = outcome_rewards_list[i]
            samples.info["reward"] = outcome_rewards_list[i]
        return samples_list, True

    # Flatten masked sequences to 2-level structure for reward model
    # Essentially list of unbatched sequences
    flattened_masked_sequences = []
    flattened_attention_mask_for_masked_sequences = []
    for i, masked_seqs in enumerate(masked_sequences_list):
        for j, seq_masked in enumerate(masked_seqs):
            if len(seq_masked) > 0:  # Has units
                flattened_masked_sequences.append(seq_masked)
                flattened_attention_mask_for_masked_sequences.append(attention_mask_for_masked_sequences_list[i][j])

    # If no masked sequences (all sequences have no units), return PPO-style rewards
    if len(flattened_masked_sequences) == 0:
        for i, samples in enumerate(samples_list):
            # Return scalar rewards for PPO compatibility
            samples.rewards = outcome_rewards_list[i]
            samples.info["reward"] = outcome_rewards_list[i]
        return samples_list, True

    # Call reward model on masked sequences
    masked_rewards_refs = reward_model_group.async_run_method_batch(
        method_name="forward",
        sequences=flattened_masked_sequences,
        attention_mask=flattened_attention_mask_for_masked_sequences,
        pad_sequence=[True] * len(flattened_masked_sequences),
    )

    # Sync to avoid GPU OOM when colocate models
    if colocate_all_models:
        ray.get(masked_rewards_refs)
        ray.get(reward_model_group.async_run_method(method_name="empty_cache"))

    # Get results and flatten nested list structure
    masked_rewards_flat = sum(ray.get(masked_rewards_refs)[::duplicate_factor], [])

    # Reconstruct rewards back to original structure
    reward_idx = 0
    for i, samples in enumerate(samples_list):
        for batch_idx in range(len(samples.sequences)):
            units = sequence_to_units_list[i][batch_idx]
            if not units:
                continue

            outcome_reward = outcome_rewards_list[i][batch_idx].item()

            # Get rewards for this sequence's masked units
            seq_masked_rewards = masked_rewards_flat[reward_idx]

            for unit_idx, (unit_start, unit_end) in enumerate(units):
                masked_reward = seq_masked_rewards[unit_idx].item()
                difference_reward = outcome_reward - masked_reward

                # Map unit indices from sequence space to action space (shift by -1)
                # Unit at token sub-sequence [start, end] maps to action [start-1, end-1]
                action_start = max(0, unit_start - 1)
                action_end = max(0, unit_end - 1)
                if action_end > action_start:
                    token_rewards_list[i][batch_idx, action_start:action_end] = drca_beta * (
                        difference_reward / (action_end - action_start)
                    )
                    print("o" * 80)
                    print("unit decoded: ", tokenizer.decode(samples.sequences[batch_idx][unit_start:unit_end]))
                    print("difference reward total: ", difference_reward)
                    print("o" * 80)

            attention_mask_seq = attention_mask_list[i][batch_idx]
            # find last element index where attention mask is one
            eos_index = attention_mask_seq.nonzero()[-1].item()
            # Map eos index to action space (shift by -1)
            eos_index = max(0, eos_index - 1)
            token_rewards_list[i][batch_idx, eos_index] = (1 - drca_beta) * (outcome_reward)

            reward_idx += 1

    # Replace outcome rewards with unit-level rewards
    for i, samples in enumerate(samples_list):
        samples.rewards = token_rewards_list[i]
        samples.info["reward"] = token_rewards_list[i]
        samples.info["outcome_reward"] = outcome_rewards_list[i]

    return samples_list, False
