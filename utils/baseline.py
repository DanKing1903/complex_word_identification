from sklearn.tree import DecisionTreeClassifier
import pyphen



class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.d = pyphen.Pyphen(lang='en')
        else:  # spanish
            self.avg_word_length = 6.2
            self.d = pyphen.Pyphen(lang='es')

        self.model = DecisionTreeClassifier(random_state=0)

    def extract_word_features(self, word, *args):
        # here are my basic features:
        # - len chars = word length
        # - len tokens = phrase length
        # - len uniq =  ratio of unique characters in word
        # - len vowels = ratio of vowels in word
        # - len const = ratio of constonants in word
        # - len syl = number of syllables
        # - final baseline system uses tokens, uniq, and const based on feature analyis

        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        len_uniq = len(set(word)) / len(word)
        len_const = len([letter for letter in word.split() if letter not in set("aeiou")]) / len(word)
        len_syl = len(self.d.inserted(word).split("-"))

        features = [len_chars, len_tokens, len_uniq, len_const, len_syl]
        return features


    def train(self, trainset, *args):
        X = []
        y = []

        for idx, sent in trainset.iterrows():
            X.append(self.extract_word_features(sent['target_word'], *args))
            y.append(sent['gold_label'])

        return self.model.fit(X, y)

    def test(self, testset, *args):
        X = []
        y = []

        for idx, sent in testset.iterrows():
            X.append(self.extract_word_features(sent['target_word'], *args))
            y.append(sent['gold_label'])

        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_

