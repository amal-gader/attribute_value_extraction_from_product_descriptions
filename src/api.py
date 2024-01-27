import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from train import config

path = os.environ.get('MODEL_PATH')
id = "LORA_2023103114"
device = "cuda:0"
model = T5ForConditionalGeneration.from_pretrained(f"{path}/models/{id}").to(device)
tokenizer = T5TokenizerFast.from_pretrained(f"{path}/models/tokenizer_{id}")

app = FastAPI()


def inference(task_prefix: str, question: str, context: str, prompt="Kontext: '{}'Frage: '{}'?, "):
    input_text = task_prefix + prompt.format(context, question)
    question_tokenized = tokenizer(input_text, max_length=config['Q_LEN'], **config['tokenizer'])
    with torch.no_grad():
        input_ids = torch.tensor(question_tokenized["input_ids"], dtype=torch.long).to(device).unsqueeze(0)
        attention_mask = torch.tensor(question_tokenized["attention_mask"], dtype=torch.long).to(device).unsqueeze(0)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        predicted_answer = tokenizer.decode(outputs.flatten(), skip_special_tokens=True)
    return predicted_answer


class InferenceRequest(BaseModel):
    question: str
    context: str
    prompt: str = "Kontext: '{}'Frage: '{}'?, "
    task_prefix: str


class Output(BaseModel):
    answer: str


@app.post("/predict", response_model=Output)
def predict(request: InferenceRequest):
    result = inference(request.task_prefix, request.question, request.context, request.prompt)
    output = Output(answer=result)
    return output
