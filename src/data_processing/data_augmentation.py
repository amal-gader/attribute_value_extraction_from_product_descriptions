import random

from googletrans import Translator
from nltk.corpus import wordnet
from nltk.corpus import words

PERIOD_TOKEN = '</s>'


class DataAugment:
    def __init__(self, text, target):
        self.text = text
        self.target = target

    @staticmethod
    def paraphrase_sentence(sentence):
        # Perform paraphrasing if possible
        paraphrased_sentence = sentence  # Default to the original sentence
        words_to_replace = sentence.split()

        for word in words_to_replace:
            # Find synonyms for each word in the sentence
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonyms = [lemma.name() for syn in synonyms for lemma in syn.lemmas()]
                synonyms = list(set(synonyms))
                synonyms = [syn for syn in synonyms if syn in words.words()]
                if synonyms:
                    synonym = random.choice(synonyms)
                    paraphrased_sentence = paraphrased_sentence.replace(word, synonym, 1)
                    break  # Only replace one word
        return paraphrased_sentence

    @staticmethod
    def backtranslate(text, target_answer, target_lang="en", source_lang="de", period=PERIOD_TOKEN):
        sentences = text.split(period)
        matching_sentences = []
        for sentence in sentences:
            if target_answer in sentence:
                matching_sentences.append(sentence)
        if len(matching_sentences) > 1:
            sentence_to_translate = random.choice(matching_sentences)
            translator = Translator()
            try:
                translated_text = translator.translate(sentence_to_translate, dest=target_lang).text
                backtranslated_text = translator.translate(translated_text, dest=source_lang).text
                text = text.replace(sentence_to_translate, backtranslated_text)
                return text

            except Exception as e:
                print(f"Translation error: {e}")
                return text
        return text

    @staticmethod
    def shuffle_sentences(text, period=PERIOD_TOKEN):
        sentences = text.split(period)
        random.shuffle(sentences)
        shuffled_text = f'{period}'.join(sentences)
        return shuffled_text

    @staticmethod
    def random_word_swap(text, target_answer, num_swaps=1, period=PERIOD_TOKEN):
        sentences = text.split(period)
        matching_sentences = []
        for sentence in sentences:
            if target_answer in sentence:
                matching_sentences.append(sentence)
        if len(matching_sentences) > 1:
            sentence_to_shuffle = random.choice(matching_sentences)
            words = sentence_to_shuffle.split()
            num_words = len(words)
            for _ in range(num_swaps):
                position1, position2 = random.sample(range(num_words), 2)
                words[position1], words[position2] = words[position2], words[position1]
            shuffled_sentence = ' '.join(words)
            text = text.replace(sentence_to_shuffle, shuffled_sentence)
        return text

    @staticmethod
    def find_and_delete_matches(text, target_answer, period=PERIOD_TOKEN):
        sentences = text.split(period)
        matching_sentences = []
        for sentence in sentences:
            if target_answer in sentence:
                matching_sentences.append(sentence)
        if len(matching_sentences) > 1:
            selected_sentence = random.choice(matching_sentences)
            matching_sentences.remove(selected_sentence)
            remaining_text = f'{period}'.join(list(set(sentences) - set(matching_sentences)))
            return remaining_text
        else:
            return text

    def random_apply(self):
        # Randomly select one of the methods to apply
        methods = [
            self.backtranslate,
            self.shuffle_sentences,
            self.random_word_swap,
            self.find_and_delete_matches
        ]
        selected_method = random.choice(methods)
        if selected_method == self.shuffle_sentences:
            return selected_method(self.text)
        else:
            text = selected_method(self.text, self.target)
            return text

