"""Reward processing hooks for custom reward computation strategies.

This module provides hooks that can be applied to reward computation
without modifying the core PPO/RLHF training code.
"""

from typing import List, Callable, Optional
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RewardHook:
    """Base class for reward processing hooks."""

    def __init__(self, args, tokenizer, reward_model_group):
        self.args = args
        self.tokenizer = tokenizer
        self.reward_model_group = reward_model_group

    def process_rewards(self, samples_list: List) -> List:
        """Process rewards in samples_list.

        Args:
            samples_list: List of Experience objects

        Returns:
            Modified samples_list
        """
        raise NotImplementedError


class UnitLevelRewardHook(RewardHook):
    """Hook for computing unit-level difference rewards."""

    def __init__(self, args, tokenizer, reward_model_group):
        super().__init__(args, tokenizer, reward_model_group)
        self.max_units = getattr(args, "max_units_per_sequence", 10)
        logger.info(f"UnitLevelRewardHook initialized with max_units={self.max_units}")

    def process_rewards(
        self,
        samples_list: List,
        drca_beta: float,
        unit_extraction: str,
        colocate_all_models: bool,
        duplicate_factor: int,
    ) -> List:
        """Replace outcome-level rewards with unit-level difference rewards.

        Args:
            samples_list: List of Experience objects with outcome rewards
            colocate_all_models: Whether all models colocated
            duplicate_factor: Duplication factor due to ring attn and tensor parallel size
            drca_beta: DRCA reward shaping parameter

        Returns:
            samples_list with unit-level rewards
            Whether no units in every sequence
        """
        from openrlhf.utils.unit_reward import compute_unit_difference_rewards_batched

        logger.info(f"Computing unit-level difference rewards for {len(samples_list)} samples")

        samples_list, no_units = compute_unit_difference_rewards_batched(
            samples_list=samples_list,
            reward_model_group=self.reward_model_group,
            tokenizer=self.tokenizer,
            drca_beta=drca_beta,
            unit_extraction=unit_extraction,
            max_units=self.max_units,
            colocate_all_models=colocate_all_models,
            duplicate_factor=duplicate_factor,
        )

        logger.info("Unit-level reward computation completed")
        return samples_list, no_units


def create_reward_hook(args, tokenizer, reward_model_group) -> Optional[RewardHook]:
    """Factory function to create reward hook based on args.

    Args:
        args: Training arguments
        tokenizer: Tokenizer
        reward_model_group: Ray actor group for reward model

    Returns:
        RewardHook instance or None if no hook enabled
    """

    return UnitLevelRewardHook(args, tokenizer, reward_model_group)


def apply_reward_hook(
    reward_hook: Optional[RewardHook],
    samples_list: List,
    drca_beta: float,
    unit_extraction: str,
    colocate_all_models: bool = False,
    duplicate_factor: int = 1,
):
    """Apply reward hook to samples if hook is enabled.

    Args:
        reward_hook: RewardHook instance or None
        samples_list: List of Experience objects
        colocate_all_models: Whether all models colocated
        duplicate_factor: Duplication factor

    Returns:
        Tuple of (samples_list, no_units)
    """
    if reward_hook is not None:
        samples_list, no_units = reward_hook.process_rewards(
            samples_list, drca_beta, unit_extraction, colocate_all_models, duplicate_factor
        )
        return samples_list, no_units
    return samples_list, True
