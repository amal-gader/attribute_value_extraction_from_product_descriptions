import datetime
import json
import os
from rich import box
from rich.table import Table, Column

training_logger = Table(Column("Epoch", justify="center"),
                        Column("Steps", justify="center"),
                        Column("Loss", justify="center"),
                        title="Training Status", pad_edge=False, box=box.ASCII)

validation_logger = Table(Column("Epoch", justify="center"),
                          Column("Steps", justify="center"),
                          Column("Loss", justify="center"),
                          Column("Accuracy", justify="center"),
                          title="Validation Status", pad_edge=False, box=box.ASCII)


class Logger:
    def __init__(self, path=None):
        self.path = path
        if self.path:
            if os.path.exists(path):
                os.remove(path)
            with open(self.path, 'w') as f:
                pass

    def log(self, msg):
        full_msg = '[{}] {}'.format(datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S'), msg)
        print(full_msg)
        if self.path:
            with open(self.path, 'w') as file:
                json.dump(full_msg, file)
