import os

import pandas as pd

from src.data_processing.data_loader import DataLoader

path = os.environ.get('DATA_PATH')
attribute_keys = ['attribute1', 'attribute2']


def dataframe_for_two_attributes():
    df = DataLoader(path).pre_process(multi_task=True)
    grouped = df.groupby(['Konzernartikelnummer', 'text']).agg({
        'attribute': list,
        'value': list
    }).reset_index()
    grouped_filtered = grouped[grouped['attribute'].apply(lambda x: len(x) >= 2)]
    df_modified = pd.DataFrame({
        'ProductID': grouped_filtered['Konzernartikelnummer'],
        'text': grouped_filtered['text'],
        'attribute1': grouped_filtered['attribute'].apply(lambda x: x[0]),
        'value1': grouped_filtered['value'].apply(lambda x: x[0]),
        'attribute2': grouped_filtered['attribute'].apply(lambda x: x[1]),
        'value2': grouped_filtered['value'].apply(lambda x: x[1])
    })
    count_threshold = 10
    df_modified['group_size'] = df_modified.groupby(attribute_keys)[attribute_keys[0]].transform('size')
    df_modified = df_modified[df_modified['group_size'] >= count_threshold].drop('group_size', axis=1)
    return df_modified


def join_attributes():
    df = DataLoader(path).pre_process()
    grouped = df.groupby(['Konzernartikelnummer', 'text']).agg({
        'attribute': list,
        'value': list
    }).reset_index()
    grouped_filtered = grouped[grouped['attribute'].apply(lambda x: len(x) >= 2)]
    df_modified = pd.DataFrame({
        'ProductID': grouped_filtered['Konzernartikelnummer'],
        'text': grouped_filtered['text'],
        'attribute': grouped_filtered['attribute'].apply(lambda x: x[0] + ', ' + x[1]),
        'value': grouped_filtered['value'].apply(lambda x: x[0] + ', ' + x[1])
    })
    return df_modified


if __name__ == '__main__':
    print(join_attributes())
