from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    vocab_size: int = 10256
    block_size: int = 256  # context size
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    per_device_train_batch_size: int = 1024
    per_device_eval_batch_size: int = 1024
    github_num_train_epochs: float = 0.1
    stackoverflow_num_train_epochs: float = 0.1
    stage1_save_steps: int = 5000
    stage2_save_steps: int = 20000

    special_tokens: Dict[str, str] = field(default_factory=dict)
    github_ds_folder: str = "github-codebase"
    stackoverflow_ds_folder: str = "stackoverflow-answers"


config = Config()
config.special_tokens = {
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
}
