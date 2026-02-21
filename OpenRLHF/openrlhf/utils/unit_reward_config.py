"""Configuration for unit-level reward computation.

This module provides configuration utilities for enabling unit-level
difference rewards in PPO training.

Usage:
    Add the following argument to your training script:

    --max_units_per_sequence 10
    
Example:
    python train_ppo_ray.py \\
        --max_units_per_sequence 10 \\
        ... (other args)
"""


def add_unit_reward_args(parser):
    """Add unit-level reward arguments to argument parser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    group = parser.add_argument_group('Unit-Level Rewards')
    
    group.add_argument(
        "--max_units_per_sequence",
        type=int,
        default=10,
        help="Maximum number of units to extract per sequence (remaining grouped as one unit)"
    )
    
    group.add_argument(
        "--unit_pattern",
        type=str,
        default="step",
        help="Pattern to identify units (currently only 'step' is supported)"
    )
    
    return parser


def validate_unit_reward_config(args):
    """Validate unit-level reward configuration.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If configuration is invalid
    """
    if args.max_units_per_sequence < 1:
        raise ValueError("max_units_per_sequence must be >= 1")
    
    if args.remote_rm_url:
        raise ValueError(
            "Unit-level rewards currently not supported with remote reward models. "
            "Please use local reward model."
        )
    
    if not args.reward_pretrain:
        raise ValueError(
            "Unit-level rewards require a reward model. "
            "Please specify --reward_pretrain"
        )
