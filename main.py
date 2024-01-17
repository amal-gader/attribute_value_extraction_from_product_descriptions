import argparse
import os

from sklearn.model_selection import train_test_split

from src.data_processing.clustering import cluster
from src.data_processing.data_loader import DataLoader as dataloader

from src.train import Trainer
from src.utils.helpers import insert_negative_samples

PATH = os.environ.get('DATA_PATH')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the training script.")
    parser.add_argument('--path', type=str, default=PATH, help="path to data")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size.")
    parser.add_argument('--multi_task',  action='store_true', help="set to false to run model on one task type.")
    parser.add_argument('--bf',  action='store_true',
                        help="Task selector, True for the Binary features task. Default is False")
    parser.add_argument('--da',  action='store_true', help="Set to true to add augmented samples")
    parser.add_argument('--insert_ng',  action='store_true', help="Set to true to add negative samples")
    parser.add_argument('--clust',  action='store_true', help="Set to true to use clustering as a subsetting method")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--epochs', type=int, default=3, help="number of training epochs")
    parser.add_argument('--full_finetune', action='store_true', help="Set to true to fully fine-tune")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.clust:
        data = cluster()
    else:
        data = dataloader(args.path, bf=args.bf).pre_process(multi_task=args.multi_task)
    if args.insert_ng:
        data = insert_negative_samples(data)
    train_data, val_data = train_test_split(data, stratify=data['attribute'], test_size=0.3, random_state=42)
    train_data, val_data = train_data.reset_index(drop=True), val_data.reset_index(drop=True)
    Trainer(train_df=train_data, val_df=val_data, device=args.device, epochs=args.epochs, full_finetune=args.full_finetune,
            batch_size=args.batch_size, insert_ad=args.da).train()


if __name__ == '__main__':
    main()
