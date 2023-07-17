from itertools import chain

from datasets import Dataset, DatasetDict
from pathlib import Path
from transformers import PreTrainedTokenizerFast

from config import config


def make_training_dataset(dataset_path: Path, tokenizer: PreTrainedTokenizerFast):
    def load_data(file_paths):
        data = []
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                text = f.read().decode()
                examples = text.split(config.special_tokens["eos_token"])
                examples = [ex + config.special_tokens["eos_token"] for ex in examples]
                data.extend(examples)
        return data


    train_data = load_data(
        [str(path) for path in dataset_path.glob("*.txt")][:-1]
    )
    validation_data = load_data(
        [str(path) for path in dataset_path.glob("*.txt")][-1:]
    )

    raw_datasets = DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_data}),
            "validation": Dataset.from_dict({"text": validation_data}),
        }
    )

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        total_length = (total_length // config.block_size) * config.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i:i + config.block_size]
                for i in range(0, total_length, config.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {config.block_size}",
    )
    return lm_datasets
