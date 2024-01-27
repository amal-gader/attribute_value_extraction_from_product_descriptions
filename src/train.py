import datetime
import json
import math
import os

import numpy as np
import torch
from pandas import DataFrame
from peft import LoraConfig, TaskType, get_peft_model, PrefixTuningConfig
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Adafactor
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers.optimization import AdafactorSchedule

from data_processing.dataset import QA_Dataset
from utils.helpers import print_number_of_trainable_model_parameters, console, insert_augmented_samples
from utils.logger import training_logger, validation_logger

PATH = os.environ.get('DATA_PATH')
new_tokens = ['®', 'Ø', '°']

with open("config.json", 'r') as config_file:
    config = json.load(config_file)
lora_config = LoraConfig(**config['lora_config'], task_type=TaskType.SEQ_2_SEQ_LM)
prefix_tuning_config = PrefixTuningConfig(**config['prefix_tuning_config'], task_type=TaskType.SEQ_2_SEQ_LM)


class Trainer:
    """
     Fine-tune T5-base for extractive and boolean question answering
     input:
     train and validation pytorch dataloaders, the chosen peft method's config, device and number of epochs
    """

    def __init__(self, train_df: DataFrame, val_df: DataFrame, peft_config=lora_config,
                 device="cuda:1", epochs=3, full_finetune=False, batch_size=16, insert_ad=False):

        self.val_df = val_df
        self.train_df = train_df
        val_dataset = QA_Dataset(self.val_df)
        train_dataset = QA_Dataset(self.train_df)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
        self.insert_ad = insert_ad
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(**self.config['model'], torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-base")
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
        if not full_finetune:
            self.model = get_peft_model(self.model, peft_config)
        self.optimizer = Adafactor(self.model.parameters(), **self.config['optimizer'])
        self.scheduler = AdafactorSchedule(self.optimizer)

    def val_loop(self, epoch, val_batch_count, val_loss):
        log_threshold = 0.05 + epoch
        epoch_val_labels = []
        epoch_val_predictions = []
        accuracy = 0
        self.model.eval()
        for batch in tqdm(self.val_loader, desc="Validation batches"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            val_loss += outputs.loss.item()
            val_batch_count += 1
            predictions, labels = self.predict(batch)
            epoch_val_labels.extend(labels)
            epoch_val_predictions.extend(predictions)
            val_progress = val_batch_count / len(self.val_loader)
            if val_progress >= log_threshold:
                eval_ppl = math.exp(val_loss / val_batch_count)
                accuracy = accuracy_score(epoch_val_predictions, epoch_val_labels)
                validation_logger.add_row(
                    str(epoch + 1),
                    str(val_batch_count),
                    f"{val_loss / val_batch_count:.4f}",
                    f"{accuracy:.4f}"
                )
                log_threshold += 0.05
                console.log(eval_ppl)
                console.print(validation_logger)
        return val_loss, val_batch_count, accuracy

    def train_loop(self, epoch, train_batch_count, train_loss):
        log_threshold = 0.05 + epoch
        self.model.train()
        for batch in tqdm(self.train_loader, desc="Training batches"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            self.optimizer.zero_grad()
            outputs.loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_loss += outputs.loss.item()
            train_batch_count += 1
            training_progress = train_batch_count / len(self.train_loader)
            if training_progress >= log_threshold:
                training_logger.add_row(
                    str(epoch + 1),
                    str(train_batch_count),
                    f"{train_loss / train_batch_count:.4f}"
                )
                console.print(training_logger)
                log_threshold += 0.05
        return train_loss, train_batch_count

    def train(self):
        train_loss = 0
        val_loss = 0
        train_batch_count = 0
        val_batch_count = 0
        console.log(print_number_of_trainable_model_parameters(self.model))
        for epoch in range(self.epochs):
            if self.insert_ad:
                self.train_df = insert_augmented_samples(self.train_df)
                self.train_df.reset_index(drop=True)
                train_dataset = QA_Dataset(self.train_df)
                self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True)
            train_loss, train_batch_count = self.train_loop(epoch, train_batch_count, train_loss)
            val_loss, val_batch_count, _ = self.val_loop(epoch, val_batch_count, val_loss)

        console.log(validation_logger)
        console.log(training_logger)
        model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
        self.model.save_pretrained(f"models/{model_name}")
        self.tokenizer.save_pretrained(f"models/tokenizer_{model_name}")
        console.save_text(f"models/logs_{model_name}.txt")
        return model_name

    def predict(self, batch):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=self.config['T_LEN']
            )
            predictions = self.tokenizer.batch_decode(outputs.to('cpu'), skip_special_tokens=True)
            labels_on_cpu = batch['labels'].cpu()
            labels = np.where(labels_on_cpu != -100, labels_on_cpu, self.tokenizer.pad_token_id)
            labels = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        return predictions, labels
