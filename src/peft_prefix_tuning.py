import os

from peft import TaskType, PrefixTuningConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data_processing.data_loader import DataLoader as dt
from src.data_processing.dataset import QA_Dataset
from train import Trainer

BATCH_SIZE = 16
DEVICE = "cuda:0"
EPOCHS = 3

peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=20)


if __name__ == '__main__':
    path = os.environ.get('DATA_PATH')
    data = dt(path).pre_process(multi_task=True)
    train_data, val_data = train_test_split(data, stratify=data['attribute'], test_size=0.3, random_state=42)
    train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    val_dataset = QA_Dataset(val_data, prefix=False)
    train_dataset = QA_Dataset(train_data, prefix=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    Trainer(train_loader, val_loader, peft_config=peft_config, device=DEVICE, epochs=EPOCHS).train()
