import sys
import os

from probability_functions.maximum_likelihood_estimator import MaxLikelihoodEstimator
from probability_functions.maximum_aposteriori import MaximumAPosteriori
from probability_functions.predictive_distribution import PredictiveDistribution
from probability_functions.prior import Prior
from text_preprocessor.metadata_extractor import DocumentLoader
from evaluation.helper_funcs import *


def get_document_words(document_name, N):
    f = open(document_name, 'r')
    full_document = [word for line in f for word in line.rstrip().split(' ')]
    return full_document[:int(N)]


def evaluate_models_perplexities(train_file, test_file, N):
    size_array = [N/128, N/64, N/16, N/4, N]
    alpha_prime = 2
    print ("Evaluating models on perplexity ... the report format is as follows")
    print ("<training size>")
    print ("MLE: training_set_perplexity - test_set_perplexity")
    print ("MAP: training_set_perplexity - test_set_perplexity")
    print ("PDE: training_set_perplexity - test_set_perplexity")

    for training_size in size_array:
        extractor = DocumentLoader(train_file=train_file, test_file=[test_file],
                                   training_size=training_size)
        metadata = extractor.get_metadata()
        prior = Prior(metadata=metadata, alpha_prime=alpha_prime).get_prior()

        MLE = MaxLikelihoodEstimator(metadata=metadata)
        MAP = MaximumAPosteriori(metadata=metadata, prior=prior)
        PDE = PredictiveDistribution(metadata=metadata, prior=prior)

        train_document = get_document_words(document_name=train_file, N=training_size)
        test_document = get_document_words(document_name=test_file, N=N)

        MLE_perplexity_train = find_perplexity(document=train_document, estimator=MLE)
        MLE_perplexity_test = find_perplexity(document=test_document, estimator=MLE)

        MAP_perplexity_train = find_perplexity(document=train_document, estimator=MAP)
        MAP_perplexity_test = find_perplexity(document=test_document, estimator=MAP)

        PDE_perplexity_train = find_perplexity(document=train_document, estimator=PDE)
        PDE_perplexity_test = find_perplexity(document=test_document, estimator=PDE)

        print ("training size: ", training_size)
        print ("MLE: ", MLE_perplexity_train, " - ", MLE_perplexity_test)
        print ("MAP: ", MAP_perplexity_train, " - ", MAP_perplexity_test)
        print ("PDE: ", PDE_perplexity_train, " - ", PDE_perplexity_test)


def evaluate_model_evidence(train_file, test_file, N):
    training_size = N/128
    print ("Evaluate prior model on evidence")
    print ("Report format: <train set perplexity> <test set perplexity> <evidence>")

    for alpha_prime in range(1, 11):
        extractor = DocumentLoader(train_file=train_file, test_file=[test_file],
                                   training_size=training_size)
        metadata = extractor.get_metadata()
        prior = Prior(metadata=metadata, alpha_prime=float(alpha_prime)).get_prior()

        PDE = PredictiveDistribution(metadata=metadata, prior=prior)

        train_document = get_document_words(document_name=train_file, N=training_size)
        test_document = get_document_words(document_name=test_file, N=N)

        PDE_perplexity_train = find_perplexity(document=train_document, estimator=PDE)
        PDE_perplexity_test = find_perplexity(document=test_document, estimator=PDE)

        evidence = find_log_evidence(metadata=metadata, prior=prior)
        print ("alpha prime: ", alpha_prime)
        print (PDE_perplexity_train, " # ", PDE_perplexity_test, " # ", evidence)


def evaluate_author_classification():
    train_file = os.path.join(os.getcwd(), "data/pg121.txt.clean")
    test_file_1 = os.path.join(os.getcwd(), "data/pg141.txt.clean")
    test_file_2 = os.path.join(os.getcwd(), "data/pg1400.txt.clean")

    meta_data = DocumentLoader(train_file=train_file, test_file=[test_file_1, test_file_2]).get_metadata()
    print (meta_data["K"])
    print (meta_data["N"])
    pass


def main(argv):
    train_file = os.path.join(os.getcwd(), "data/training_data.txt")
    test_file = os.path.join(os.getcwd(), "data/test_data.txt")

    N = 640000 # 640000 is the total number of words in the training data

    # evaluate_models_perplexities(train_file=train_file, test_file=test_file, N=N)
    # evaluate_model_evidence(train_file, test_file, N)
    evaluate_author_classification()

if __name__ == "__main__":
    main(sys.argv[1:])