from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

from config import config


ds = load_dataset(
    "codeparrot/github-code", streaming=True, split="train", languages=["Python"]
)

#####
"""
{
 'code': "import mod189 from './mod189';\nvar value=mod189+1;\nexport default value;\n",
 'repo_name': 'MirekSz/webpack-es6-ts',
 'path': 'app/mods/mod190.js',
 'language': 'JavaScript',
 'license': 'isc',
 'size': 73
}
"""

output_folder = Path(config.github_ds_folder)
output_folder.mkdir(exist_ok=True)

part = 0
output_file = output_folder / f"gihub_cb_part_{part}.txt"
f = output_file.open("wb")
for i, element in enumerate(tqdm(ds), 1):
    string = (
        f'repo_name: {element["repo_name"]}\n'
        f'path: {element["path"]}\n'
        f'code:\n{element["code"]}\n'
        f'{config.special_tokens["eos_token"]}'
    )
    # string = string.replace('\n', '\\n')
    string = string.encode("utf-8")
    f.write(string)

    if i % 1000 == 0:
        f.close()
        part += 1
        output_file = output_folder / f"github_cb_part_{part}.txt"
        f = output_file.open("wb")
f.close()
