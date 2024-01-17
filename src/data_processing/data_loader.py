import logging
import os

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.utils.caching import cache_data
from src.utils.helpers import is_value_in_text, normalize, format_text

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
path = os.environ.get('DATA_PATH')


def rep_set(df, k=20):
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    df = df.query("`text_length`>5").pipe(lambda _df: _df[_df.groupby('attribute')['attribute'].transform('size') >= k])
    df, _ = train_test_split(df, test_size=0.4, stratify=df['attribute'], random_state=42)
    return df


def unseen_data(df, k=20):
    return df.pipe(lambda _df: _df[_df.groupby('attribute')['attribute'].transform('size') < k])


class DataLoader:
    def __init__(self, data_path, bf=False):
        self.data_path = data_path
        self.bf = bf
        self.irrelevant_columns = ['Purpose', 'ECLASS_8_1', 'ECLASS_Name', 'Marke', 'Hersteller', 'text_length']
        self.irrelevant_attributes = ['PBS-Marke', 'PBS-OEM Nummer', 'PBS-Variantenattribut',
                                      'PBS-Inhalt', 'Grundtext', 'Lieferanten-Artikelnummer',
                                      'Hersteller-Artikelnummer', 'Lieferantenartikelnummer',
                                      'Herstellerartikelnummer', 'Kurz-Nr.']

    @property
    @cache_data
    def descriptions(self):
        logger.info("Loading descriptions data...")
        df = DataFrame()
        for filename in os.listdir(self.data_path):
            if "Artikelbeschreibung" in filename:
                df = pd.concat([df, pd.read_excel(f"{self.data_path}/{filename}", engine="openpyxl")])
        return df

    @property
    @cache_data
    def attributes(self):
        logger.info("Loading attributes data...")
        df = DataFrame()
        for filename in os.listdir(self.data_path):
            if "Artikelattribute" in filename:
                df = pd.concat([df, pd.read_excel(f"{self.data_path}/{filename}", engine="openpyxl")])
        return df

    @property
    @cache_data
    def merged_data(self):
        """
        Merge the different dataframes in order to keep products with a description and labels
        """
        descriptions = self.descriptions
        attributes = self.attributes
        merged = attributes.merge(descriptions, on='Konzernartikelnummer', how='right')
        merged.rename(columns={"AttributeValue": "value", "AttributeKey": "attribute", "Text": "text"}, inplace=True)
        return merged

    def pre_process(self, multi_task=False):
        """
           Pre-processes the merged data for the Question Answering model.
           Args:
               multi_task (bool, optional): If True, includes multi-task filtering. Default is False.
           """

        def task_specific_filtering(df, bf):
            """
                    Filters the DataFrame based on the specified task.
            """
            df["value"] = df["value"].apply(normalize)
            if bf:
                return df.query(f"value in ['true', 'false']")
            elif multi_task:
                return df[
                    df.apply(
                        lambda row:
                        True if row["value"] in ['false', 'true'] else is_value_in_text(row['value'], row['text']),
                        axis=1
                    )
                ]
            else:
                return df[
                    ~df["value"].apply(lambda x: str(x).lower()).isin(['false', 'true'])
                    & df.apply(lambda x: is_value_in_text(x['value'], x['text']),
                               axis=1
                        )
                    ]

        return (
            self.merged_data.query(f'attribute not in {self.irrelevant_attributes}')
            .drop_duplicates(subset=['attribute', 'text', 'value'])
            .dropna(subset=['value', 'attribute', 'text'])
            .astype({"attribute": str, "value": str, "text": str})
            .pipe(task_specific_filtering, self.bf)
            .assign(
                **{
                    'text': lambda df: df.text.apply(format_text),
                    'value': lambda df: df.value.apply(lambda x: format_text(x, skip_eos=True))
                }
            )
            .query('~value.str.strip().eq("") & (value.str.strip() != "</s>")')
            .pipe(rep_set)
            .drop(columns=self.irrelevant_columns)
            .reset_index(drop=True)
        )


if __name__ == '__main__':
    path = os.environ.get('DATA_PATH')
    data_loader = DataLoader(path)
    dframe = data_loader.pre_process().pivot_table(columns='attribute', aggfunc='size')
