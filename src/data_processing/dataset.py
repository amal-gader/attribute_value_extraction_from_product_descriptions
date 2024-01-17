import torch
import json
from torch.utils.data import Dataset
from transformers import T5TokenizerFast

TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
new_tokens = ['®', 'Ø', '°']
TOKENIZER.add_tokens(new_tokens)

with open("../config.json", 'r') as config_file:
    config = json.load(config_file)

prefixes = {'bf': "Beantworten Sie die folgende Frage mit true oder false. \n",
            'ov': "Beantworten die folgende Frage basierend auf dem Kontext. \n"}


class QA_Dataset(Dataset):
    """
       Custom PyTorch dataset for Question-Answer pairs.
       Args:
           dataframe : Input DataFrame containing columns 'attribute', 'text', and 'value'.
           prefix (bool, optional): Whether to include task-specific prefixes. Default is True
       """

    def __init__(self, dataframe, prefix=True):
        self.prefix = prefix
        self.data = dataframe
        self.questions = self.data["attribute"]
        self.context = self.data["text"]
        self.answer = self.data["value"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]
        return self.encode(answer, question, context, prefix=self.prefix)

    @staticmethod
    def encode(answer, question, context, prefix, prompt="Kontext: '{}'Frage: '{}'?, "):
        """
                Encodes the answer, question, and context into tokenized inputs.
                Returns:
                    dict: Tokenized inputs and labels.
         """
        task_prefix = ''
        if prefix:
            if answer in ['false', 'true']:
                task_prefix = prefixes['bf']
            else:
                task_prefix = prefixes['ov']

        input_text = task_prefix + prompt.format(context, question)
        question_tokenized = TOKENIZER(input_text, max_length=config['Q_LEN'], **config['tokenizer'])
        answer_tokenized = TOKENIZER(answer, max_length=config['T_LEN'], **config['tokenizer'])
        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100

        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
        }
