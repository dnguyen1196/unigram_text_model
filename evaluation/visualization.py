import matplotlib.pyplot as plt
import os


def plot_perplexities_vs_training_size(model_perplexities):
    training_size = [tup[0] for tup in model_perplexities[0]]
    MLE_train = [tup[1] for tup in model_perplexities[0]]
    MLE_test = [tup[2] for tup in model_perplexities[0]]

    MAP_train = [tup[1] for tup in model_perplexities[1]]
    MAP_test = [tup[2] for tup in model_perplexities[1]]

    PDE_train = [tup[1] for tup in model_perplexities[2]]
    PDE_test = [tup[2] for tup in model_perplexities[2]]


    l1, = plt.plot(training_size, MLE_train, label="MLE perplexity (train)", marker='o', markersize=5)
    l2, = plt.plot(training_size, MAP_train, label="MAP perplexity (train)", marker='o', markersize=5)
    l3, = plt.plot(training_size, PDE_train, label="PDE perplexity (train)", marker='o', markersize=5)
    plt.xlabel('training size')
    plt.ylabel('perplexity of train data')
    plt.legend([l1,l2,l3], ['MLE', 'MAP', 'PDE'])
    plt.title('Variation of training size on train set perplexity')
    plt.savefig(os.path.join(os.getcwd(), 'results/train_perplexity_vs_training_size'))
    plt.close()


    l1, = plt.plot(training_size, MLE_test, label="MLE perplexity (test)", marker='o', markersize=10)
    l2, = plt.plot(training_size, MAP_test, label="MAP perplexity (test)", marker='o', markersize=5)
    l3, = plt.plot(training_size, PDE_test, label="PDE perplexity (test)", marker='o', markersize=5)
    plt.xlabel('training size')
    plt.ylabel('perplexity of test data')
    plt.legend([l1,l2,l3], ['MLE', 'MAP', 'PDE'])
    plt.title('Variation of training size on test set perplexity')
    plt.savefig(os.path.join(os.getcwd(), 'results/test_perplexity_vs_training_size'))
    plt.close()


def plot_evidence_against_alpha(model_evidence):
    alpha_array = [tup[0] for tup in model_evidence]
    perplexity_array = [tup[1] for tup in model_evidence]
    evidence_array = [tup[2] for tup in model_evidence]


    plt.figure(figsize=(8,6))
    plt.subplot(2, 1, 1)
    plt.plot(alpha_array, perplexity_array, label="test data perplexity", marker='o', markersize=5)
    plt.xlabel('alpha')
    plt.ylabel('perplexity of test data')

    plt.subplot(2, 1, 2)
    plt.plot(alpha_array, evidence_array, label="log evidence", marker='o', markersize=5)
    plt.xlabel('alpha')
    plt.ylabel('log evidence')

    plt.savefig(os.path.join(os.getcwd(), 'results/evidence_perplexity_vs_alpha'))
    plt.close()

