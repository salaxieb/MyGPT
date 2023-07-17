from pathlib import Path
from transformers import PreTrainedTokenizerFast
import typer
from transformers import AutoModelForCausalLM, GPT2Config
from safetensors.torch import save_model
from transformers import Trainer
from transformers import TrainingArguments
from safetensors.torch import save_model, load_model

from train_dataset import make_training_dataset
from config import config


app = typer.Typer()


def load_tokenizer(path: Path = Path("tokenizer") / "gpt-code.json"):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path))
    tokenizer.add_special_tokens(config.special_tokens)
    return tokenizer


def init_or_load_model(tokenizer: PreTrainedTokenizerFast, save_path: Path):
    configuration = GPT2Config()

    configuration.n_embd = config.n_embd
    configuration.n_head = config.n_head
    configuration.n_layer = config.n_layer
    configuration.n_positions = config.block_size  # context size
    configuration.vocab_size = tokenizer.vocab_size
    configuration.bos_token_id = tokenizer.bos_token_id
    configuration.eos_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_config(configuration)
    if save_path.exists():
        load_model(model, save_path)
    return model


@app.command()
def train(dataset: str, continue_from_checkpoint: bool = False):
    assert dataset in [
        config.github_ds_folder,
        config.stackoverflow_ds_folder,
    ], "unknown dataset"
    tokenizer = load_tokenizer()
    lm_datasets = make_training_dataset(Path(dataset), tokenizer)
    save_path: Path = Path("trained_model") / "codebase_GPT.safetensors"
    model = init_or_load_model(tokenizer, save_path)

    checkpoints_path = Path("trained_model").resolve()

    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        do_train=True,
        do_eval=True,
        output_dir=str(checkpoints_path),
        num_train_epochs=config.github_num_train_epochs,
        fp16=True,
        save_safetensors=True,
        save_steps=config.stage1_save_steps,
        logging_steps=config.stage1_save_steps,
        optim="adamw_torch",  # torch.optim.AdamW,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
    )

    resume_from_checkpoint = False
    if continue_from_checkpoint and list(checkpoints_path.glob("*")):
        resume_from_checkpoint = checkpoints_path
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    save_model(model, save_path)


if __name__ == "__main__":
    app()
