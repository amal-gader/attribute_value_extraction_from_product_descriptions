from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AttributeClassifier:
    def __init__(self, df: DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer()
        self.decision_stump = DecisionTreeClassifier(max_depth=10)

    def preprocess_data(self):
        # TODO find a better input representation
        product_descriptions = self.df.text
        # + 'Attribut: ' + self.df.attribute
        attribute_values = self.df.value.to_list()
        attribute_values = [1 if value else 0 for value in attribute_values]
        X = self.vectorizer.fit_transform(product_descriptions)
        y = np.array(attribute_values)
        return X, y

    def train_model(self, X_train: DataFrame, y_train: DataFrame):
        return self.decision_stump.fit(X_train, y_train)

    def _predict(self, X_test):
        return self.decision_stump.predict(X_test)

    @staticmethod
    def evaluate_model(y, y_pred) -> dict:
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1}

    def predict(self, description: str) -> str:
        description_vectorized = self.vectorizer.transform([description])
        predicted_attribute_value = self.decision_stump.predict(description_vectorized)
        return predicted_attribute_value[0]
