import logging
import os

import random
from pathlib import Path
import spacy
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from spacy.scorer import Scorer
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split
from src.utils.helpers import has_overlap
from src.data_processing.data_transformations import dataframe_for_two_attributes
from tqdm import tqdm

path = os.environ.get('DATA_PATH')
dir = '/output/'
logger = logging.getLogger(__name__)
attribute_keys = ['attribute1', 'attribute2']
cols = [('attribute1', 'value1'), ('attribute2', 'value2')]
description_column = "text"
attribute_value_column = "value1"


class CustomNerTrainer:
    def __init__(self, df_train: DataFrame, df_test: DataFrame, output_dir: str = dir):
        self.df_train = df_train
        self.df_test = df_test
        self.output_dir = Path(output_dir)
        self.nlp = spacy.blank("de")
        self.nlp.add_pipe('ner')
        self.ner = self.nlp.get_pipe('ner')
        self.other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        self.sgd = self.nlp.create_optimizer()

    #  TODO: get rid of for loops
    def prepare_data(self, df: DataFrame) -> list[Example]:
        data = []
        for _, row in df.iterrows():
            text = row[description_column]
            entities = []
            #  TODO retrain using one label per training sample
            for key, att in cols:
                attribute = row[att]
                label = row[key]
                start_index = text.lower().find(attribute.lower())
                if start_index != -1:
                    end_index = start_index + len(attribute)
                    entities.append((start_index, end_index, label))
            if all(not has_overlap(ent1[:2], ent2[:2]) for i, ent1 in enumerate(entities) for j, ent2 in
                   enumerate(entities) if i != j):
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, {"entities": entities})
                data.append(example)
        return data

    def train_custom_ner(self):
        training_data = self.prepare_data(self.df_train)
        self.nlp.initialize()
        all_losses = []
        with self.nlp.disable_pipes(*self.other_pipes):
            for iteration in tqdm(range(30)):
                random.shuffle(training_data)
                losses = {}
                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    self.nlp.update(list(batch), losses=losses, drop=0.1, sgd=self.sgd)
                logger.info("epoch: {} Losses: {}".format(iteration, str(losses)))
                all_losses.append(losses['ner'])

    def save_model(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        self.nlp.meta['name'] = 'custom_ner'
        self.nlp.to_disk(self.output_dir)
        logger.info("Saved model to", self.output_dir)

    def evaluate(self, samples):
        scorer = Scorer(self.nlp)
        example = []
        for sample in samples:
            pred = self.nlp(sample['text'])
            print(pred, sample['entities'])
            temp_ex = Example.from_dict(pred, {'entities': sample['entities']})
            example.append(temp_ex)
        scores = scorer.score(example)
        return scores

    #  TODO add F1 score
    def eval(self, df, label):
        def extract_prediction(text, label, model):
            doc = model(text)
            entity_labels = [ent.label_ for ent in doc.ents]
            if label not in entity_labels:
                return None
            return next((ent.text for ent in doc.ents if ent.label_ == label), '')

        df = df.assign(
            prediction=lambda _df: _df[description_column].apply(extract_prediction, label=label, model=self.nlp))
        df['prediction'].fillna('', inplace=True)
        y = df[attribute_value_column]
        y_pred = df['prediction']
        accuracy = accuracy_score(y, y_pred)
        return accuracy


if __name__ == '__main__':
    df = dataframe_for_two_attributes()
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df[attribute_keys])
    trainer = CustomNerTrainer(train_data, test_data)
    trainer.train_custom_ner()
    trainer.save_model()
    trainer.evaluate(test_data)
