from datasets import load_dataset
from pathlib import Path

from config import config

ds = load_dataset("koutch/stackoverflow_python", split="train", streaming=True)

#####
"""
{
    title (string)
    question_id (int64)
    question_body (string)
    question_score (int64)
    question_date (string)
    answer_id (int64)
    answer_body (string)
    answer_score (int64)
    answer_date (string)
    tags (sequence)
}
"""

output_folder = Path(config.stackoverflow_ds_folder)
output_folder.mkdir(exist_ok=True)

part = 0
output_file = output_folder / f"stackoverflow_answers_part_{part}.txt"
f = output_file.open("wb")
for i, element in enumerate(ds, 1):
    string = (
        f'title: {element["title"]}\n'
        f'question: {element["question_body"]}\n'
        f'answer:\n{element["answer_body"]}\n'
        f'answer_score: {element["answer_score"]}\n'
        f'{config.special_tokens["eos_token"]}'
    )
    # string = string.replace('\n', '\\n')
    string = string.encode("utf-8")
    f.write(string)

    if i % 1000 == 0:
        f.close()
        part += 1
        output_file = output_folder / f"stackoverflow_answers_part_{part}.txt"
        f = output_file.open("wb")
f.close()
