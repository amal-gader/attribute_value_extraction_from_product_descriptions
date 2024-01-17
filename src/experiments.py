import argparse
import json
import os
import logging

import pandas

from src.data_processing.data_loader import DataLoader
from modeling import T5, Bert
from src.data_processing.data_transformations import join_attributes

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATH = os.environ.get('DATA_PATH')
dict = {"t5": T5(), "bert": Bert()}


def first_experiment(data_path=PATH, bf=False):
    data_loader = DataLoader(data_path, bf=bf)
    df = data_loader.pre_process()
    df = df.drop_duplicates(subset=['AttributeKey'])
    logger.info("Starting results generation...")
    model = T5()
    model.infer(df)
    with open('results/metrics', 'w') as file:
        json.dump(model.eval(df), file)
    df.to_csv("oneshot.csv", encoding='utf-16', index=False)


def sentence_selection(data_path=PATH, bf=False):
    data_loader = DataLoader(data_path, bf)
    df = data_loader.pre_process()
    logger.info("Starting results generation...")
    model = T5()
    model.infer(df, bf)
    with open('results/metrics', 'w') as file:
        json.dump(model.eval(df), file)
    df.to_csv("oneshot.csv", encoding='utf-16', index=False)


# TODO: Add the category name in the metrics file
def few_shots(data_path=PATH, bf=False, model_name='t5'):
    data_loader = DataLoader(data_path, bf)
    df = data_loader.pre_process()
    df = df.groupby('AttributeKey')
    result = pandas.DataFrame()
    model = dict[model_name]
    logger.info("Starting results generation...")
    excel_filename = f"results/fewshots_{model_name}_{'bf' if bf else 'ovf'}.xlsx"
    json_filename = f"results/metrics_{model_name}_{'bf' if bf else 'ovf'}.json"
    with open(json_filename, 'w') as file:
        for category, sub_df in df:
            sub_df = sub_df.head(10)
            logger.info(f"Category: {category} ....")
            model.infer(sub_df, bf=bf)
            metrics = model.eval(sub_df, bf=bf)
            json.dump(metrics, file)
            file.write('\n')
            result = result.append(sub_df)
    result.to_excel(excel_filename, engine="openpyxl", index=False)


def prompt_engineering(data_path=PATH, bf=False):
    prompts = {'bf': ["'{}'\nBased on that paragraph can we conclude that this sentence is true?\n'{}'",
                      "'{}'\nbased on the context is the following property valid?\n Property:'{}'",
                      "'{}'\nBased on the context answer this yes/no question\n'{}'?",
                      "Read the following paragraph and determine if the hypothesis is true:\n'{}'\nHypothesis: "
                      "'{}'"],
               'ov': ["Kontext: '{}'\n Frage: '{}'?",
                      "Kontext: '{}'\nBeantworten die folgende Frage basierend auf dem Kontext\n Frage: '{}'",
                      "Kontext: '{}' Extrahieren Sie aus dem Kontext den Attributwert für: '{}' "]
               }
    if bf:
        prompts = prompts['bf']
    else:
        prompts = prompts['ov']
    data_loader = DataLoader(data_path, bf=bf)
    df = data_loader.pre_process()
    model = T5()
    metrics_all_prompts = {}
    for i, prompt in enumerate(prompts):
        logger.info(f"Prompt: {prompt}")
        model.infer(df, prompt)
        metrics = model.eval(df)
        metrics_all_prompts.update({f"Prompt_{i + 1}": metrics})
        df.to_csv(f"Prompt_{i + 1}.csv", encoding='utf-16', index=False)
    with open('metrics_all_prompts.json', 'w') as file:
        json.dump(metrics_all_prompts, file)


def one_sample_inference(context, question):
    model = T5()
    print(model.get_attribute_value(context, question))


def inference_for_two_attributes():
    df = join_attributes()
    df = df.groupby('AttributeKey')
    result = pandas.DataFrame()
    logger.info("Starting results generation...")
    model = dict['t5']
    csv_filename = f"results/fewshots_multiple_attribute_query.csv"
    json_filename = f"../results/metrics_multiple_attribute_query.json"

    with open(json_filename, 'w') as file:
        for category, sub_df in df:
            sub_df = sub_df.head(10)
            logger.info(f"Category: {category} ....")
            model.infer(sub_df, bf=False)
            metrics = model.eval(sub_df, bf=False)
            json.dump(metrics, file)
            file.write('\n')
            result = result.append(sub_df)
    result.to_csv(csv_filename, encoding='utf-16', index=False)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument('--experiment', type=str, choices=['first', 'second', 'third', 'fourth'], default='first',
                        help="Select the experiment to run (first or second). Default is 'first'.")
    parser.add_argument('--data_path', type=str, default=PATH,
                        help="Path to the data for the experiments. Default is PATH.")
    parser.add_argument('--bf', type=bool, default=False,
                        help="Task selector, True for the Binary features task. Default is False")
    parser.add_argument('--model', type=str, default='t5', help="Model name to run")
    parser.add_argument('--context', type=str, default='',
                        help="Give a product description from which you want to extract attributes")
    parser.add_argument('--question', type=str, default='', help="That's the attribute label to extract")

    args = parser.parse_args()
    if args.experiment == 'first':
        first_experiment(args.data_path, args.bf)
    elif args.experiment == 'second':
        few_shots(args.data_path, args.bf, args.model)
    elif args.experiment == 'third':
        one_sample_inference(args.context, args.question)
    elif args.experiment == 'fourth':
        inference_for_two_attributes()
