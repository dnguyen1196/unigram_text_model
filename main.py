"""
Main program that calls on all the functions to intialize model, do evaluation
"""

import sys

from probability_functions.maximum_likelihood_estimator import MaxLikelihoodEstimator
from probability_functions.maximum_aposteriori import MaximumAPosteriori
from probability_functions.predictive_distribution import PredictiveDistribution
from probability_functions.prior import Prior
from text_preprocessor.metadata_extractor import DocumentExtractor
from evaluation.helper_funcs import *
from evaluation.visualization import *

#
# Helper function to perform evaluation of model perplexities
#
def evaluate_models_perplexities(train_file, test_file, N):
    size_array = [N/128, N/64, N/16, N/4, N]
    alpha_prime = 2

    print("Evaluating models on perplexity ... the report format is as follows")
    print("<training size>")
    print("<estimator>: <training_set_perplexity> # <test_set_perplexity>")

    # Since they all use the same test set
    test_document = get_document_words(document_name=test_file, N=N)
    model_perplexities = [[], [], []]

    for training_size in size_array:
        extractor = DocumentExtractor(train_file=train_file, test_file=[test_file],
                                      training_size=training_size)
        metadata = extractor.get_metadata()
        prior = Prior(metadata=metadata, alpha_prime=alpha_prime).get_prior()

        MLE = MaxLikelihoodEstimator(metadata=metadata)
        MAP = MaximumAPosteriori(metadata=metadata, prior=prior)
        PDE = PredictiveDistribution(metadata=metadata, prior=prior)

        train_document = get_document_words(document_name=train_file, N=training_size)

        MLE_perplexity_train = find_perplexity(document=train_document, estimator=MLE)
        MLE_perplexity_test = find_perplexity(document=test_document, estimator=MLE)

        MAP_perplexity_train = find_perplexity(document=train_document, estimator=MAP)
        MAP_perplexity_test = find_perplexity(document=test_document, estimator=MAP)

        PDE_perplexity_train = find_perplexity(document=train_document, estimator=PDE)
        PDE_perplexity_test = find_perplexity(document=test_document, estimator=PDE)

        model_perplexities[0].append((int(training_size), MLE_perplexity_train, MLE_perplexity_test))
        model_perplexities[1].append((int(training_size), MAP_perplexity_train, MAP_perplexity_test))
        model_perplexities[2].append((int(training_size), PDE_perplexity_train, PDE_perplexity_test))

        print()
        print("training size: ", int(training_size))
        print("MLE: ", MLE_perplexity_train, " - ", MLE_perplexity_test)
        print("MAP: ", MAP_perplexity_train, " - ", MAP_perplexity_test)
        print("PDE: ", PDE_perplexity_train, " - ", PDE_perplexity_test)

    return model_perplexities


#
# Helper function to perform evaluation of model evidence
#
def evaluate_model_evidence(train_file, test_file, N):
    training_size = N/128 # Training size = N/128
    print("Evaluate prior model on evidence")
    print("Report format: <test set perplexity> # <evidence>")

    extractor = DocumentExtractor(train_file=train_file, test_file=[test_file], training_size=training_size)
    metadata = extractor.get_metadata()

    model_evidence = []

    test_document = get_document_words(document_name=test_file, N=N)

    for alpha_prime in range(1, 11):
        prior = Prior(metadata=metadata, alpha_prime=float(alpha_prime)).get_prior()

        PDE = PredictiveDistribution(metadata=metadata, prior=prior)
        PDE_perplexity_test = find_perplexity(document=test_document, estimator=PDE)
        evidence = find_log_evidence(metadata=metadata, prior=prior)

        model_evidence.append((alpha_prime, PDE_perplexity_test, evidence))

        print("Alpha prime: ", alpha_prime)
        print(PDE_perplexity_test, " # ", evidence)

    return model_evidence


#
#   Helper function to classify author based on text perplexity
#
def evaluate_author_classification(train_file, test_file_1, test_file_2):
    alpha_prime = 2
    print("Evaluating perplexity for author classification")

    document_perplexities = []

    # Train
    metadata_1 = DocumentExtractor(train_file=train_file, test_file=[test_file_1, test_file_2]).get_metadata()
    prior_1 = Prior(metadata=metadata_1, alpha_prime=alpha_prime).get_prior()
    PDE = PredictiveDistribution(metadata=metadata_1, prior=prior_1)

    test_document_1 = get_document_words(test_file_1)
    test_document_2 = get_document_words(test_file_2)

    # Find perplexity
    perplexity_1 = find_perplexity(test_document_1, PDE)
    perplexity_2 = find_perplexity(test_document_2, PDE)
    document_perplexities.append((perplexity_1, perplexity_2))
    print("Without removing infrequent words")
    print("pg141: ", perplexity_1)
    print("pg1400: ", perplexity_2)

    # Retrain while removing infrequent words
    metadata_2 = DocumentExtractor(train_file=train_file, test_file=[test_file_1, test_file_2], min_word=50).get_metadata()
    prior_2 = Prior(metadata_2, alpha_prime=alpha_prime).get_prior()
    PDE = PredictiveDistribution(metadata=metadata_2, prior=prior_2)

    # Find perplexity
    perplexity_1 = find_perplexity(test_document_1, PDE)
    perplexity_2 = find_perplexity(test_document_2, PDE)
    document_perplexities.append((perplexity_1, perplexity_2))
    print()
    print("After removing infrequent words")
    print("pg141: ", perplexity_1)
    print("pg1400: ", perplexity_2)

    return document_perplexities


#
# Entry into the program
#
def main(argv):
    train_file = os.path.join(os.getcwd(), "training_data.txt")
    test_file = os.path.join(os.getcwd(), "test_data.txt")
    total_word_count = 640000

    model_perplexities = evaluate_models_perplexities(train_file=train_file, test_file=test_file, N=total_word_count)
    plot_perplexities_vs_training_size(model_perplexities)

    model_evidence = evaluate_model_evidence(train_file, test_file, N=total_word_count)
    plot_evidence_against_alpha(model_evidence=model_evidence)

    train_file = os.path.join(os.getcwd(), "pg121.txt.clean")
    test_file_1 = os.path.join(os.getcwd(), "pg141.txt.clean")
    test_file_2 = os.path.join(os.getcwd(), "pg1400.txt.clean")
    evaluate_author_classification(train_file, test_file_1, test_file_2)


if __name__ == "__main__":
    main(sys.argv[1:])