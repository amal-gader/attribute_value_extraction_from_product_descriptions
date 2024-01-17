from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')


def find_k_closest_sentences(text, answer, k=4, period='<eos>'):
    sentences = text.split(period)
    sentence_embeddings = model.encode(sentences)
    answer_embedding = model.encode([answer])
    cosine_similarities = util.pytorch_cos_sim(answer_embedding, sentence_embeddings)[0]
    top_indices = cosine_similarities.argsort(descending=True)[:k]
    closest_sentences = [sentences[i] for i in top_indices]
    concatenated_sentences = ' '.join(closest_sentences)
    return concatenated_sentences


if __name__ == '__main__':
    text = 'format a 4 <eos> ' \
           'stabiler karton kaschiert mit naturbelassenem kraftpapier <eos>' \
           ' greener work komplett kunststofffrei unverpackt mit papiereinleger recycelbar <eos>' \
           ' auch zum aufhaengen <eos> ' \
           'ausfuehrung klemme material der klemme metall verchromt anordnung der lage klemme kurze seite max <eos>' \
           ' klemmdicke 8 mm hellsilber <eos> ' \
           'ausfuehrung klemmbrett max <eos> ' \
           'anzahl der blaetter 80 verwendung fuer papiergroessen a 4 min <eos> ' \
           'dicke 19 mm farbe blau <eos> groesse b x h 233 mm x 32.3 mm <eos>'
    query = 'Farbe'
    print(find_k_closest_sentences(text, query, k=4))
