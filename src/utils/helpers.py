import random
import time
import re
import pandas as pd
from rich.console import Console
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from data_processing.data_augmentation import DataAugment

PERIOD_TOKEN = '</s>'
console = Console(record=True)


def post_processing(x):
    x = x.replace("<pad>", '').replace("</s>", '').strip()
    return x


def process_space_insensitive(text):
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def format_text(text, skip_eos=False):
    period_exceptions = ['inkl', 'incl', 'ca', 'zb']

    text_formatted = ' '
    text = text + ' '
    for i, ch in enumerate(text[:-1]):

        # if regular character ('a', '3')
        if ch.isalnum():
            if text_formatted and ((text_formatted[-1].isalpha() and ch.isnumeric()) or
                                   (text_formatted[-1].isnumeric() and ch.isalpha())):
                text_formatted = text_formatted + ' ' + (ch.lower())
            else:
                text_formatted += ch.lower()

        # if space (' ')
        elif ch == ' ':
            if not text_formatted[-1] == ' ':
                text_formatted += ' '

        elif ch in ['.', ',']:

            # if period or comma ('a.', '3,7')
            if text_formatted[-1].isnumeric() and text[i + 1].isnumeric():
                text_formatted += '.'

            # abbreviation with period
            elif ch == '.' and text[i + 1] == ' ' and any(
                    [text_formatted.lower().endswith(x) for x in period_exceptions]):
                text_formatted += '.'

            # if end of sentence
            elif ch == '.' and text[i + 1] == ' ':
                text_formatted += ' ' + PERIOD_TOKEN

        # if special character separating regular characters ('a-b')
        elif text_formatted[-1].isalnum() and text[i + 1].isalnum() and not ch in ['\n', '\r']:
            text_formatted += ' '

        elif ch == ';' or ch in ['\n', '\r']:
            pass

    if not skip_eos and not text_formatted.endswith(PERIOD_TOKEN):
        text_formatted += (' ' + PERIOD_TOKEN)

    return text_formatted[1:]


def create_negative_sample(text, answer):
    answer = str(answer)
    text = str(text)
    # Create a regular expression pattern to match the answer case-insensitively
    pattern = re.compile(re.escape(answer), re.IGNORECASE)
    negative_sample = pattern.sub("", text)

    return negative_sample


def insert_negative_samples(df):
    df_copy = df[~df["value"].apply(lambda x: str(x).lower()).isin(['false', 'true'])]
    num_samples_to_select = int(0.2 * len(df_copy))
    selected_samples = df_copy.sample(n=num_samples_to_select, random_state=42)
    tqdm.pandas(desc="Negative Samples", position=0, leave=True)
    selected_samples['text'] = selected_samples.progress_apply(
        lambda row: create_negative_sample(row['text'], row['value']), axis=1)
    selected_samples['value'] = 'Keine Antwort'
    df = pd.concat([df, selected_samples], ignore_index=True)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    return shuffled_df


def normalize(x):
    value_mapping = {
        'yes': 'true',
        'no': 'false',
        'ja': 'true',
        'nein': 'false',
        'true': 'true',
        'false': 'false'
    }
    return value_mapping.get(x.lower(), x)


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")

    return wrapper


def is_value_in_text(value, text):
    value = str(value).lower()
    text = text.lower()
    if value in text:
        return True
    return False


def has_overlap(span1, span2):
    start1, end1 = span1
    start2, end2 = span2
    return not (end1 <= start2 or end2 <= start1)


dimension_pattern = r'(\d+(?:[,.]\d+)?)(?:\s*[xX]\s*(\d+(?:[,.]\d+)?))(?:\s*(?:[a-zA-Z]+)\b)'


def extract_dimensions(description):
    match = re.search(dimension_pattern, description)
    if match:
        return match.group(0)
    return None


patterns = [
    (1, r'breite\D*(\d+(?:[,.]\d+)?)\s*(?:[cm]*\b)'),
    (2, r'Größe\D*\D*?(\d+(?:[,.]\d+)?)\s*[xX]\s*(\d+(?:[,.]\d+)?)\s*[a-zA-Z]*'),
    (3, r'Größe\D*\D*?(\d+(?:[,.]\d+)?)\s*(?:[a-zA-Z]*\b)'),
    (4, r'Größe \(B x H x T\):\s*(\d+)')
]

default_pattern = r'\b\d+(?:[,.]\d+)?(?:\s*[xX]\s*\d+(?:[,.]\d+)?){1,2}(?:\s*[a-zA-Z]*\b)'


def extract_width(description):
    for priority, pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            return match.group(1)
    # If no rule matches, use the default pattern
    match = re.findall(default_pattern, description)
    if match:
        return match[0]
    return None


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def insert_augmented_samples(df):
    num_samples_to_generate = int(len(df) * 0.1)

    def augment_data(text, target):
        try:
            data_augment = DataAugment(text, target)
            augmented_text = data_augment.random_apply()
            return augmented_text
        except Exception as e:
            console.log(f"Data augmentation failed for text: {text}, target: {target}. Error: {str(e)}")

    random_indices = random.sample(range(len(df)), num_samples_to_generate)
    subset_df = df.iloc[random_indices]
    tqdm.pandas(desc="Augmenting Data", position=0, leave=True)
    subset_df['text'] = subset_df.progress_apply(lambda x: augment_data(x['text'], x['value']), axis=1)
    df = pd.concat([df, subset_df], ignore_index=True)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    # df = df.pipe(insert_negative_samples)
    return shuffled_df


def split_train_val_test(df):
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    train_df, temp_df = train_test_split(df, stratify=df['attribute'], test_size=1 - train_ratio, random_state=42)
    val_df, test_df = train_test_split(temp_df, stratify=temp_df['attribute'],
                                       test_size=test_ratio / (test_ratio + val_ratio), random_state=42)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
