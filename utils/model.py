from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from utils.transformers import Selector, Affix_Extractor, WordFeatureExtractor
import pyphen




class Model(object):

    def __init__(self, language, model = "rfc"):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.d = pyphen.Pyphen(lang='en')
        else:  # spanish
            self.avg_word_length = 6.2
            self.d = pyphen.Pyphen(lang='es')

        if model == "dtc":
            self.model = DecisionTreeClassifier(class_weight = "balanced", max_depth = 100, max_features = 100, random_state=0)
        elif model == "rfc":
            self.model = RandomForestClassifier(n_estimators=50, max_depth=50, max_features = 100,random_state=0, class_weight="balanced")
        self.build_pipe()

    def build_pipe(self):
        word_features = Pipeline([('select', Selector(key="target_word"))] +
            [('wfe', WordFeatureExtractor(self.language))] +
            [('dv', DictVectorizer() )])

        Ngrams = Pipeline([('select', Selector(key="sentence"))] +
            [('cv', CountVectorizer())])

        suffix = Pipeline([('select', Selector(key='target_word'))]+
                          [('suf',Affix_Extractor(affix_type = 'suffix'))]+
                          [( 'dv', DictVectorizer())])

        prefix = Pipeline([('select', Selector(key='target_word'))]+
                          [('suf',Affix_Extractor(affix_type = 'prefix'))]+
                          [( 'dv', DictVectorizer())])

        char_ngrams = Pipeline([('select', Selector(key = 'target_word'))]+
                              [('cv', CountVectorizer(analyzer = 'char_wb',ngram_range=(2, 3)))])       


        f_union = Pipeline([('union', FeatureUnion(transformer_list = [
                            ('words', word_features),('ngrams', Ngrams),
                            ('sffx', suffix), ('pffx', prefix),
                            ('char', char_ngrams)]))])


        self.pipe = f_union

    def train(self, trainset, *args):
        X = self.pipe.fit_transform(trainset)
        y = trainset['gold_label']

        return self.model.fit(X, y)

    def test(self, testset, *args):
        X = self.pipe.transform(testset)
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_
