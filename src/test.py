import datetime
import os

import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from src.data_processing.dataset import prefixes
from evaluation import Evaluator
from train import config

from utils.helpers import console
from src.data_processing.data_loader import DataLoader


class Test:
    """
    Evaluate a trained model
    Args:
        df (DataFrame): Test data
        model_id: name of the model to test
        device: cuda device
        metrics: a list of evaluation metrics, if None EM and F1 scores are calculated
        save_predictions (bool): Set to true to save model predictions
    """
    def __init__(self, df: DataFrame, model_id: str, device="cuda:1", metrics=None, save_predictions=False):
        self.df = df
        self.device = device
        self.metrics = metrics
        self.id = model_id
        self.save_predictions = save_predictions
        self.model = T5ForConditionalGeneration.from_pretrained(f"models/{self.id}").to(self.device)
        self.tokenizer = T5TokenizerFast.from_pretrained(f"models/tokenizer_{self.id}")

    def inference(self, answer: str, question: str, context: str, prompt="Kontext: '{}'Frage: '{}'?, "):
        if answer in ['false', 'true']:
            task_prefix = prefixes['bf']
        else:
            task_prefix = prefixes['ov']
        input_text = task_prefix + prompt.format(context, question)
        question_tokenized = self.tokenizer(input_text, max_length=config['Q_LEN'], **config['tokenizer'])
        with torch.no_grad():
            input_ids = torch.tensor(question_tokenized["input_ids"], dtype=torch.long).to(self.device).unsqueeze(0)
            attention_mask = torch.tensor(question_tokenized["attention_mask"], dtype=torch.long).to(
                self.device).unsqueeze(0)
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
            predicted_answer = self.tokenizer.decode(outputs.flatten(), skip_special_tokens=True)
        return predicted_answer

    def test(self):
        tqdm.pandas(desc="Inference", position=0, leave=True)
        self.df['prediction'] = self.df.progress_apply(
            lambda x: self.inference(x['value'], x['attribute'], x['text']), axis=1)
        try:
            Evaluator().compute_metrics(self.df, self.metrics)
        except Exception as e:
            console.log(f"Evaluation Failed. Error: {str(e)}")
        if self.save_predictions:
            self.df.to_excel(f"predictions_{self.id}.xlsx")
        console.save_text(f"models/test_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.txt")
        return self.df


if __name__ == '__main__':
    path = os.environ.get('DATA_PATH')
    data = DataLoader(path).pre_process(multi_task=True)
    train_data, val_data = train_test_split(data, stratify=data['attribute'], test_size=0.3, random_state=42)
    train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    val_data = val_data.query("`attribute`=='Anzahl der Etiketten je Blatt'")
    id = "LORA_2023103114"
    test = Test(val_data, id).test()
