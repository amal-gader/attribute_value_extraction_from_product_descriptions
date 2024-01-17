import evaluate
import nltk
from pandas import DataFrame
from rouge_score import rouge_scorer

from utils.helpers import console

predicted_col_label = 'prediction'
ground_truth_col_label = 'value'


class Evaluator:
    @staticmethod
    def calculate_exact_match(df: DataFrame, bf=False):
        def exact_match(ground_truth: str, prediction: str):
            if bf:
                return int(prediction == ground_truth)
            return int(prediction.lower() == ground_truth.lower())

        df['EM'] = df.apply(lambda x: exact_match(x[predicted_col_label], x[ground_truth_col_label]), axis=1)

    @staticmethod
    def calculate_f1_score(df: DataFrame):
        def f1_score(ground_truth: str, prediction: str):
            predicted_tokens = set(nltk.word_tokenize(prediction))
            ground_truth_tokens = set(nltk.word_tokenize(ground_truth))
            common_tokens = set(predicted_tokens) & set(ground_truth_tokens)
            precision = len(common_tokens) / len(predicted_tokens) if len(predicted_tokens) > 0 else 0
            recall = len(common_tokens) / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        df['F1'] = df.apply(lambda x: f1_score(x[predicted_col_label], x[ground_truth_col_label]), axis=1)

    @staticmethod
    def calculate_rouge_metric(df: DataFrame):
        def rouge(answer: str, prediction: str):
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(answer, prediction)
            result = scores['rouge1']
            return result

        df['ROUGE'] = df.apply(lambda x: rouge(x[predicted_col_label], x[ground_truth_col_label]), axis=1)

    @staticmethod
    def calculate_bleu_metric(df: DataFrame):
        def bleu(answer: str, prediction: str):
            bleu = evaluate.load("google_bleu")
            result = bleu.compute(predictions=[prediction], references=[[answer]])
            return result['google_bleu']

        df['Bleu'] = df.apply(lambda x: bleu(x[predicted_col_label], x[ground_truth_col_label]), axis=1)

    def compute_metrics(self, df: DataFrame, metrics: list[str] = None) -> DataFrame:
        if metrics is None:
            metrics = ['F1', 'EM']

        if 'F1' in metrics:
            self.calculate_f1_score(df)
            console.log(df['F1'].mean())

        if 'EM' in metrics:
            self.calculate_exact_match(df)
            console.log(df['EM'].mean())

        if 'Bleu' in metrics:
            self.calculate_bleu_metric(df)
            console.log(df['Bleu'].mean())

        return df
