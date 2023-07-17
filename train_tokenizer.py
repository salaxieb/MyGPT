from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import normalizers
from tokenizers.normalizers import NFKD, Lowercase, StripAccents, NFD
# from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import pre_tokenizers
from tokenizers.processors import ByteLevel
from tokenizers.trainers import BpeTrainer

from config import config

if __name__ == '__main__':
    gpt_tokenizer = Tokenizer(BPE(unk_token=config.special_tokens['unk_token']))
    gpt_tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
    gpt_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    gpt_tokenizer.post_processor = ByteLevel()


    trainer = BpeTrainer(
        vocab_size=config.vocab_size,  # (int, optional) — The size of the final vocabulary, including all tokens and alphabet.
        show_progress=True,  # (bool, optional) — Whether to show progress bars while training.
        special_tokens=list(config.special_tokens.values()) + ['\n'],  # (List[Union[str, AddedToken]], optional) — A list of special tokens the model should know of.
        continuing_subword_prefix="##",  # (str, optional) — A prefix to be used for every subword that is not a beginning-of-word.
        max_token_length=50,  # (int, optional) — Prevents creating tokens longer than the specified size. This can help with reducing polluting your vocabulary with highly repetitive tokens like ====== for wikipedia
    )

    files = [str(file) for file in Path(config.github_ds_folder).glob("*.txt")]
    files += [str(file) for file in Path(config.stackoverflow_ds_folder).glob("*.txt")]

    gpt_tokenizer.train(files, trainer)

    Path("tokenizer").mkdir(exist_ok=True)
    gpt_tokenizer.save("tokenizer/gpt-code.json")


    # with open('github-codebase\gihub_cb_part_0.txt', 'rb') as f:
    #     code_snippet = f.read().decode().split('[EOS]')[0]

    # print(gpt_tokenizer.encode(code_snippet).ids)
    # print(gpt_tokenizer.decode(gpt_tokenizer.encode(code_snippet).ids))