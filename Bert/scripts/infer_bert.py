import os
import yaml
from dataclasses import dataclass, field
from typing import Optional

import datasets

from transformers import (
    HfArgumentParser,
    pipeline,
    TrainingArguments
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

def main():
    with open('./tc_transformers.yaml', 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)
    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    training_args, model_args = parser.parse_args_into_dataclasses()

    timestamp = "1700461341"
    output_dir = os.path.abspath((os.path.join(os.path.curdir, training_args.output_dir, timestamp)))

    if params['task'] == "SST2":
        eval_dataset = datasets.load_dataset("glue", "sst2", split="validation", cache_dir=model_args.cache_dir)
        sentence = eval_dataset['sentence'][0]
    elif params['task'] == "COLA":
        eval_dataset = datasets.load_dataset("glue", "cola", split="validation", cache_dir=model_args.cache_dir)
        sentence = eval_dataset['sentence'][0]
    elif params['task'] == "MR":
        pass

    inference = pipeline('text-classification', model=output_dir, tokenizer=output_dir)
    print('*' * 5, 'Inference Results', '*' * 5)
    print(inference(sentence))



if __name__ == "__main__":
    main()