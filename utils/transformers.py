# adapted from
# https://opendevincode.wordpress.com/2015/08/01/building-a-custom-python-scikit-learn-transformer-for-machine-learning/
# and http://michelleful.github.io/code-blog/2015/06/20/pipelines/
# http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py

from sklearn.base import BaseEstimator, TransformerMixin
import pyphen


class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, *_):
        return self

    def transform(self, df):
        return df[self.key]


class Affix_Extractor(BaseEstimator, TransformerMixin):
    def __init__(self, affix_type):
        self.affix_type = affix_type

    def transform(self, X, *_):
        result = []
        for word in X:
            if self.affix_type == "prefix":
                affix = word[:3]
            elif self.affix_type == "suffix":
                affix = word[-3:]

            row_dict = {affix: 1}
            result.append(row_dict)
        return result

    def fit(self, *_):
        return self


class WordFeatureExtractor(BaseEstimator, TransformerMixin):
    # here are my basic features:
        # - len chars = word length
        # - len tokens = phrase length
        # - len uniq =  ratio of unique characters in word
        # - len vowels = ratio of vowels in word
        # - len const = ratio of constonants in word
        # - len syl = number of syllables
        # - final baseline system uses tokens, uniq, and const based on feature analyis

    def __init__(self, language):
        language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.d = pyphen.Pyphen(lang='en')

    def transform(self, X, *_):
        result = []
        for word in X:
            len_chars = len(word) / self.avg_word_length
            len_tokens = len(word.split(' '))
            len_uniq = len(set(word)) / len(word)
            len_const = len([letter for letter in word.split() if letter not in set("aeiou")]) / len(word)
            len_syl = len(self.d.inserted(word).split("-"))

            # dictionary to store the features in, in order to access later when testing individual features
            row_dict = {"chars": len_chars, "tokens": len_tokens, "unique": len_uniq, "const": len_const, "syl": len_syl}

            result.append(row_dict)
        return result

    def fit(self, *_):
        return self
