# Unit-Level Rewards for Fine-Grained Credit Assignment

This document describes how to use unit-level difference rewards in OpenRLHF's PPO training.

## Overview

Unit-level rewards provide fine-grained credit assignment by computing difference rewards for each "unit" (e.g., step) in the generated output, rather than assigning a single outcome-level reward to the entire sequence.

### Difference Reward Formula

For each unit `i` in the output:

```
R_unit_i = R_msg(x, y) - R_msg(x, y_without_unit_i)
```

Where:
- `R_msg(x, y)` is the outcome-level reward for the full output
- `R_msg(x, y_without_unit_i)` is the reward when unit `i` is masked out
- Tokens not in any unit receive the outcome-level reward `R_msg(x, y)`

## How It Works

1. **Unit Extraction**: After generation, each output is parsed to identify "units" (steps)
   - Units are identified by step labels: `A.`, `B.`, `AA.`, `1.`, `*1.`, etc.
   - Maximum of 10 units per sequence (remaining grouped as one)

2. **Difference Reward Computation**: For each unit:
   - Create a modified sequence with the unit masked (replaced with spaces)
   - Compute reward for the modified sequence
   - Difference reward = original reward - modified reward

3. **Per-Token Reward Assignment**:
   - Tokens within unit `i` receive `R_unit_i`
   - Tokens outside any unit receive `R_msg(x, y)`

4. **PPO Training**: The per-token rewards flow through standard PPO advantage computation

## Usage

### 1. Enable Unit-Level Rewards

Add these arguments to your training command:

```bash
python openrlhf/cli/train_ppo_ray.py \
    --use_unit_level_rewards \
    --max_units_per_sequence 10 \
    --reward_pretrain <path_to_reward_model> \
    ... (other standard PPO args)
```

### 2. Configuration Options

- `--use_unit_level_rewards`: Enable unit-level rewards (default: False)
- `--max_units_per_sequence`: Maximum units to extract (default: 10)
- `--unit_pattern`: Pattern for unit identification (default: "step")

### 3. Requirements

- Local reward model (not compatible with `--remote_rm_url`)
- Reward model specified via `--reward_pretrain`

## Implementation Details

### Files Added

1. `openrlhf/utils/unit_extraction.py`: Unit extraction logic
2. `openrlhf/utils/unit_reward.py`: Difference reward computation
3. `openrlhf/utils/reward_hooks.py`: Hook system for reward processing
4. `openrlhf/utils/unit_reward_config.py`: Configuration utilities

### Integration Point

The unit-level reward computation is applied as a hook in the experience maker, after outcome-level rewards are computed but before advantage calculation. This design ensures:

- No modification to core PPO algorithm
- Easy to enable/disable via command-line flag
- Compatible with existing PPO features

### Computational Cost

- Each sequence with N units requires N+1 reward model forward passes
- Modified sequences are batched for efficiency
- Expect ~N× increase in reward model computation time

## Example Output Format

The system expects outputs in this format:

```
A. Is the product a UV Light unit? If YES, go to Step B. If NO, APPROVE.
B. Does the product make a sterilization claim? If YES, RESTRICT. If NO, APPROVE.
```

Each step starts with a label (`A.`, `B.`, etc.) at the beginning of a line.

## Monitoring

The following metrics are logged:
- Number of units extracted per sequence
- Unit extraction success rate
- Difference reward statistics

## Troubleshooting

**Issue**: No units extracted from outputs
- **Solution**: Check that your model generates outputs with step labels (`A.`, `B.`, etc.)

**Issue**: Out of memory during training
- **Solution**: Reduce `--rollout_batch_size` or `--micro_rollout_batch_size` to account for additional reward model calls

**Issue**: Training is very slow
- **Solution**: This is expected due to N× reward model calls. Consider using fewer units or smaller reward model.

## References

- Hierarchical credit assignment in RL
- REINFORCE with fine-grained rewards
- PPO with per-token rewards
