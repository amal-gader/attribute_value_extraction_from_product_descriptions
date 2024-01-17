import os
import pandas as pd
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.data_processing.data_loader import DataLoader


def is_duplicate(text1, text2, threshold=50):
    return fuzz.ratio(text1, text2) > threshold


path = os.environ.get('DATA_PATH')
data = DataLoader(path).pre_process(multi_task=True)
texts = data['text'].tolist()

# unique_texts = [texts[0]]
# for text in texts[1:]:
#     if not any(is_duplicate(text, unique_text) for unique_text in unique_texts):
#         unique_texts.append(text)
#
# print(unique_texts)

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
embeddings = model.encode(texts)
print(len(texts))
unique_texts = [texts[0]]
for i in tqdm(range(1, len(texts)), desc="deduplicate"):
    similarities = cosine_similarity([embeddings[i]], embeddings[:i])
    if max(similarities[0]) < 0.8:
        unique_texts.append(texts[i])

print(len(unique_texts))
df = pd.DataFrame({'Texts': unique_texts})
df.to_excel('unique_text.xlsx', index=False)


