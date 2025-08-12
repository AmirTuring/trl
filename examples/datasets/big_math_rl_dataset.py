from typing import Any, Optional
from datasets import Dataset, load_dataset


def format_math(
    data: dict[str, str | float | int], output_key: str = "answer_correct"
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data[output_key],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_big_math_rl_dataset(
    split: str = "train",
    seed: int = 42,
    test_size: float = 0.05,
    output_key: str = "answer_correct",
) -> dict[str, Dataset | None]:
    """Load and split the Big-Math-RL-filtered dataset into train and validation sets using HF's train_test_split."""
    print(
        "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    # Load the original dataset
    original_ds = load_dataset("AmirMohseni/Big-Math-RL-filtered", split=split)

    # Split into train and validation sets using HF's train_test_split
    split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

    # Format the examples, removing original columns
    train_formatted = split_ds["train"].map(
        format_math,
        remove_columns=split_ds["train"].column_names,
        fn_kwargs={"output_key": output_key},
    )
    val_formatted = split_ds["test"].map(
        format_math,
        remove_columns=split_ds["test"].column_names,
        fn_kwargs={"output_key": output_key},
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class BigMathRLDataset:
    def __init__(
        self,
        split: str = "train",
        seed: int = 42,
        test_size: float = 0.05,
        output_key: str = "answer_correct",
    ):
        """Initialize the BigMathRL dataset with train/validation split.

        Args:
            split: Dataset split to use
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0)
            output_key: Key for the correct answer column
            prompt_file: Optional prompt file path
        """
        self.formatted_ds = prepare_big_math_rl_dataset(
            split=split, seed=seed, test_size=test_size, output_key=output_key
        )
