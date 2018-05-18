from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from utils.model import Model

import random


def execute_demo(language, model):
    data = Dataset(language)

    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])
    if model == "Baseline":
        model = Baseline(language)

    elif model == "Decision Tree Classifier":
        model = Model(language, "dtc")

    elif model == "Random Forest Classifier":
        model = Model(language, "rfc")

    model.train(data.trainset)

    predictions = model.test(data.testset)

    gold_labels = data.testset['gold_label']

    report_score(gold_labels, predictions)


if __name__ == '__main__':
    random.seed(42)

    for model in ["Baseline", "Decision Tree Classifier", "Random Forest Classifier"]:
        print(model)
        execute_demo('english', model)
        execute_demo('spanish', model)


