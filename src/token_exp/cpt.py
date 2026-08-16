from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, default_data_collator, TrainerCallback, set_seed
from datasets import load_dataset, concatenate_datasets
import torch
from token_exp.configs import *
import os
import random
import numpy as np




def run_cpt(model, train_dataset, output_dir = "cpt_dir", peak_lr = 3e-5, warmup_ratio = 0.03, epochs = 1,
              tokenizer=None, resume_from_checkpoint=None, callbacks=None, seq_len : int =256):
    
    os.makedirs(output_dir, exist_ok=True)

    def tokenize(batch):
        texts = [(text or "") + tokenizer.eos_token for text in batch["Text"]]
        tokens = tokenizer(
            texts,               
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        input_ids = torch.tensor(tokens["input_ids"])
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        tokens["input_ids"] = input_ids
        tokens["labels"] = labels
        return tokens

    train_dataset = train_dataset.map(tokenize , batched = True)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=epochs,
        learning_rate=peak_lr,
        lr_scheduler_type="cosine",
        #warmup_ratio=warmup_ratio,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_total_limit=1,
        save_steps=500,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        #tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model = trainer.model
    trainer.save_model('./full_model')
    tokenizer.save_pretrained('./full_model')
    return model , tokenizer



