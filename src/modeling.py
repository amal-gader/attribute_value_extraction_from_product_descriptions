import logging

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForQuestionAnswering

from evaluation import Evaluator
from utils.helpers import time_function, post_processing, normalize

logger = logging.getLogger(__name__)


class Model:
    @time_function
    def infer(self, df):
        pass

    @staticmethod
    def eval(df, bf=False):
        evaluator = Evaluator()
        metrics = None
        if bf:
            metrics = ['EM']
        return evaluator.compute_metrics(df, metrics)


# TODO: Review attributes and method parameters definition
class T5(Model):
    def __init__(self):
        self.checkpoint = 'google/flan-t5-xxl'
        self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint)
        self.tokenizer.add_tokens(['®', '²', 'Ø', 'à', 'é', 'è', '°', '´', 'á'])
        self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint)
        # load_in_8bit=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to('cuda')
        # load_in_8bit=True, device_map="auto"), torch_dtype=torch.float8 device_map="auto",

    def get_attribute_value(self, context, question, prompt="Kontext: '{}'Frage: '{}'?, "):
        input_text = prompt.format(context, question)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0])

    @time_function
    def infer(self, df, prompt="Extrahieren die Attributewert für:'{}', vom Kontext: '{}'", bf=False):
        if bf:
            df['prediction'] = df.apply(
                lambda x: normalize(
                    post_processing(
                        self.get_attribute_value(x['Text'], x['AttributeKey'], prompt="'{}'\nbased on the context is the following property valid?\n Property:'{}'"))), axis=1)
        else:
            df['prediction'] = df.apply(lambda x: post_processing(self.get_attribute_value(x['Text'], x['AttributeKey'], prompt=prompt)), axis=1)


class Bert(Model):
    def __init__(self):
        self.checkpoint = "deutsche-telekom/bert-multi-english-german-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.tokenizer.add_tokens(['®', '²', 'Ø', 'à', 'é', 'è', '°', '´', 'á'])
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.checkpoint)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def infer(self, df, bf=False):
        def get_attribute_value(context, question):
            input = self.tokenizer(question, context, padding=True, truncation=True, return_tensors="pt")
            output = self.model(**input)
            start_logit, end_logit = output.start_logits, output.end_logits
            answer_start = torch.argmax(start_logit, dim=1)
            answer_end = torch.argmax(end_logit, dim=1) + 1
            answer = [self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input["input_ids"][0][answer_start[0]:answer_end[0]]))]
            return answer[0]

        df['prediction'] = df.apply(lambda x: get_attribute_value(x['Text'], x['AttributeKey']), axis=1)
