#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
import warnings
import logging
import math
import os
import numpy as np
import csv
import sys
import torch
import wandb
import glob
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm 

from invariant_trainer import InvariantTrainer

from invariant_roberta import InvariantRobertaForMaskedLM, InvariantRobertaConfig
from invariant_distilbert import InvariantDistilBertForSequenceClassification, InvariantDistilBertConfig

import transformers
from transformers import (
    CONFIG_MAPPING,
    TOKENIZER_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

CONFIG_MAPPING.update({'invariant-distilbert': InvariantDistilBertConfig})
CONFIG_MAPPING.update({'invariant-roberta': InvariantRobertaConfig})

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.update({InvariantDistilBertConfig: InvariantDistilBertForSequenceClassification})
MODEL_FOR_MASKED_LM_MAPPING.update({InvariantRobertaConfig: InvariantRobertaForMaskedLM})

TOKENIZER_MAPPING.update({InvariantDistilBertConfig: (DistilBertTokenizer, DistilBertTokenizerFast)})
TOKENIZER_MAPPING.update({InvariantRobertaConfig: (RobertaTokenizer, RobertaTokenizerFast)})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    mode: Optional[str] = field(
        default="ilm",
        metadata={
            "help": "Whether to train the heads as an ensemble instead of following the IRM-games dynamics"}
    )
    nb_steps_heads_saving: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps between saving the head weights (if 0, the heads are not saved regularly)."},
    )
    nb_steps_model_saving: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps between saving the full model (if 0, the heads are not saved regularly)."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file or a directory)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input validation data file or directory (a text file or directory)."},
    )
    ood_validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Fichier de validation Out-Of-Distribution (.txt) jamais vu à l'entraînement"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether to treat each line of the input files as a separate document."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    nb_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of training steps."}
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Aucun fichier d'entraînement ni dataset n'a été spécifié.")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    On surcharge la classe par défaut pour permettre la boucle d'entraînement IRM Games.
    """
    head_updates_per_encoder_update : Optional[int] = field(
        default=1,
        metadata={"help": "Number of head updates per encoder update (IRM Games)"}
    )



def compute_metrics(p: EvalPrediction):
    """
    Calcule accuracy, precision, recall et f1 (moyenne pondérée)
    à partir des prédictions et labels du Trainer.
    """
    # p.predictions a la forme (batch_size, num_labels)
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.local_rank != -1:
        torch.cuda.set_device(training_args.local_rank)

    if is_main_process(training_args.local_rank):
        wandb.init(
            project="Comparaison",
            name=training_args.run_name,
            config=training_args.to_dict()
        )
           
    nb_steps = data_args.nb_steps

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    train_folder = data_args.train_file
    datasets = {}
    # Chargement des datasets d'entraînement
    if training_args.do_train:
        if train_folder is not None:  
            for file in os.listdir(train_folder):
                if file.endswith('.txt'):
                    env_name = file.split(".")[0]
                    data_files = {"train": os.path.join(train_folder, file)}
                    datasets[env_name] = load_dataset(
                        "csv",
                        data_files=data_files,
                        delimiter="\t",
                        column_names=["text", "labels"],
                    )
        else:
            raise ValueError("Aucun fichier d'entraînement ni dataset n'a été spécifié.")

    # Chargement de la validation depuis le dossier "val_env"
    if training_args.do_eval:
        if data_args.validation_file is not None:
            data_files = {"validation": data_args.validation_file}
            datasets["ind-validation"] = load_dataset(
                "csv",
                data_files=data_files,
                delimiter="\t",
                column_names=["text", "labels"],
            )
        else:
            raise ValueError("Aucun fichier de validation n'est spécifié pour l'évaluation.")

    # Configuration du modèle et du tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir, # répertoire où HuggingFace stock les fichiers téléchargés (tokenizer, vocab, etc.).
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    #tokenizer.add_special_tokens({"additional_special_tokens": ["🔴", "🟢"]})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<AAA>", "<BBB>"]})
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
 
    envs = [k for k in datasets.keys() if 'validation' not in k]
    
    if 'envs' not in config.to_dict():
        if model_args.model_name_or_path:
            inv_config = InvariantDistilBertConfig(envs=envs, num_labels=28, **config.to_dict())
            irm_model = InvariantDistilBertForSequenceClassification(inv_config, model)
        else:
            raise ValueError("Modèle inconnu")
    else:
        irm_model = model
    
    irm_model.resize_token_embeddings(len(tokenizer))

    # Pré-traitement des datasets d'entraînement : tokenisation et regroupement.
    irm_tokenized_datasets = {}
    for env_name, datasets in datasets.items():
        if training_args.do_train and 'validation' not in env_name:
            column_names = datasets["train"].column_names
        elif training_args.do_eval and 'validation' in env_name:
            column_names = datasets["validation"].column_names
        text_column_name = "content" if "content" in column_names else column_names[0]
        
        # Calcul de max_seq_length pour l'entraînement
        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warn(f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). Picking 1024 instead.")
                max_seq_length = 1024
        else:
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
            
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # 1) Remove empty lines et Tokenisation du texte uniquement
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            tokenized = tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

            # 2) On ajoute la colonne "labels" provenant du dataset
            tokenized["labels"] = examples["labels"]
            return tokenized

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
        irm_tokenized_datasets[env_name] = tokenized_datasets

    DatasetDict(irm_tokenized_datasets).save_to_disk("datas/irm_tokenized_datasets")
    
    for env_name, tokenized_ds in irm_tokenized_datasets.items():
        tokenized_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

    # Data collator pour MLM
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)
    data_collator = DataCollatorWithPadding(tokenizer)
    
    train_tokenized_datasets = {k: v for k, v in irm_tokenized_datasets.items() if 'ind-validation' not in k}
    eval_ind_tokenized_datasets = irm_tokenized_datasets['ind-validation']['validation']

    # Initialisation du Trainer
    trainer = InvariantTrainer(
        model=irm_model,
        args=training_args,
        eval_dataset=eval_ind_tokenized_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        if last_checkpoint is not None:
            check_point = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            check_point = model_args.model_name_or_path
        else:
            check_point = None
            warnings.warn("No checkpoint found. Training from scratch.")
        
        if model_args.mode == "ilm":
            train_result = trainer.invariant_train(
                training_set=train_tokenized_datasets,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                resume_from_checkpoint=check_point
            )

        elif model_args.mode == "game":
            train_result = trainer.invariant_train_games(
                training_set=train_tokenized_datasets,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                resume_from_checkpoint=check_point
            )

        output_dir = training_args.output_dir
        trainer.model.save_pretrained(output_dir, safe_serialization=False) # sauvegarde le modèle
        tokenizer.save_pretrained(output_dir) # sauvegarde le tokenizer
    
        if trainer.is_world_process_zero() and wandb.run:
            wandb.finish()
        if trainer.is_world_process_zero() and training_args.do_eval:
            wandb.init(
                project="Comparaison",
                name=f"{training_args.run_name}-eval",
                config=training_args.to_dict(),
                reinit=True
            )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        print(
            f"Accuracy: {metrics['eval_accuracy']:.4f}, "
            f"F1 (weighted): {metrics['eval_f1']:.4f}"
        )
        if trainer.is_world_process_zero():
            wandb.log({
                "eval/accuracy": metrics["eval_accuracy"],
                "eval/f1": metrics["eval_f1"],
            })

    if wandb.run:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
