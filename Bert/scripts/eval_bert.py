import logging
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
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
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments, ModelArguments))
    data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    with open('../config/tc_transformers.yaml', 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    timestamp = "1718763141"
    output_dir = os.path.abspath((os.path.join(os.path.curdir, training_args.output_dir, timestamp)))
    data_params = params['data_files']

    if params['task'] == "SST2":
        eval_dataset = datasets.load_dataset("glue", "sst2", split="validation", cache_dir=model_args.cache_dir)
    elif params['task'] == "COLA":
        eval_dataset = datasets.load_dataset("glue", "cola", split="validation", cache_dir=model_args.cache_dir)
    elif params['task'] == "MR":
        data_files = {}
        data_files["test"] = data_params['test_file']
        eval_dataset = datasets.load_dataset("csv", data_files=data_files, split="test")


    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir, config=config)
    model.to(device)

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["sentence"],
            padding=padding,
            max_length=params["max_seq_length"],
            truncation=True,
        )

    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("*** Predict ***")

    predictions, labels, metrics = trainer.predict(eval_dataset, metric_key_prefix="predict")

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()