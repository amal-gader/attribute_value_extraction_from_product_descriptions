import datetime
import os

from pandas import DataFrame
from sklearn.model_selection import KFold
from tqdm import tqdm

from train import Trainer
from utils.helpers import console, print_number_of_trainable_model_parameters
from src.data_processing.data_loader import DataLoader as dataloader
kf = KFold(n_splits=3, random_state=42, shuffle=True)
# kf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)


def train_with_kfold(data: DataFrame, batch_size=16):
    """
        use the Trainer class applying k-fold cross-validation.
        Args:
            data: DataFrame containing the training data with columns 'attribute', 'text', and 'value'.
            batch_size (int, optional): Batch size for training. Default is 16.
        Returns:
            str: Name of the best-trained model.
    """
    fold = 0
    acc = {}
    best_accuracy = 0
    accuracy = 0
    # for train_index, val_index in tqdm(kf.split(data, data['attribute']), desc="Folds"):
    for train_index, val_index in tqdm(kf.split(data), desc="Folds"):
        fold += 1
        train_df, val_df = data.iloc[train_index], data.iloc[val_index]
        train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
        trainer = Trainer(train_df, val_df)
        train_loss = 0
        val_loss = 0
        train_batch_count = 0
        val_batch_count = 0
        console.log(print_number_of_trainable_model_parameters(trainer.model))
        for epoch in range(trainer.epochs):
            train_loss, train_batch_count = trainer.train_loop(epoch, train_batch_count, train_loss)
            val_loss, train_batch_count, accuracy = trainer.val_loop(epoch, val_batch_count, val_loss)
        acc[fold] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state_dict = trainer.model.state_dict()

    best_model_name = f"{trainer.model.peft_type}_best_model_{datetime.datetime.now().strftime('%Y%m%d%H')}"
    trainer.model.load_state_dict(best_model_state_dict)
    trainer.model.save_pretrained(best_model_name)
    trainer.tokenizer.save_pretrained(f"tokenizer_{best_model_name}")
    console.save_text(f"logs_{best_model_name}.txt")

    return best_model_name


if __name__ == '__main__':
    path = os.environ.get('DATA_PATH')
    data = dataloader(path).pre_process(multi_task=True)
    train_with_kfold(data)
