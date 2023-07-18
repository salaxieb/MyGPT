setup:
	pip install -r requirements.txt
	DS_BUILD_OPS=1 pip install deepspeed
	ds_report


dataset:
	python github_codebase_dataset.py
	python stackoverflow_dataset.py


tokenizer:
	python train_tokenizer.py


train_stage1:
	python train.py github-codebase


train_stage2:
	python train.py stackoverflow-answers
