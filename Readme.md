# Python GPT 
### this project is done for training purposes, don't use it production

# GPT trained on:
## Stage.1: 
### 100 x H100 GPU hours
trained on Python suset of [github-codebase corpus](codeparrot/github-code)

## stage.2:
### 20 x H100 GPU hours
finetuned on [Stackoverflow Python answers subset](koutch/stackoverflow_python)

## inference
tbd


## Train yourself:
```
make dataset
make tokenizer
make train_stage1
make train_stage2
```

todo:
infer
upload/download to hf