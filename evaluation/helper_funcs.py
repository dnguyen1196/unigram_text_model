"""
Function to calculate the log evidence based on the dictionary obtained from
the document and the given prior
"""

import math

"""
the formula for the evidence function is P(Data | alpha)
= gamma(a0) [gamma(a1+m1)...gamma(aK+mK)] / gamma(a0 + N) [gamma(a1)...gamma(aK)]
"""


def find_log_evidence(metadata, prior):
    n = metadata["N"]
    a0 = prior["a0"]
    dictionary = metadata["dictionary"]

    log_evidence = 0.0
    for i in range(int(a0), int(a0 + n)):
        log_evidence -= math.log(i)

    for word in dictionary:
        if dictionary[word] >= 0:
            mk = dictionary[word]
            ak = prior[word]
            for j in range(int(ak), int(ak + mk)):
                log_evidence += math.log(j)

    return log_evidence


def find_perplexity(document, estimator):
    """
    document is an array of words
    For each word in the document, call on the estimator to find the probability
    of a word
    """
    log_probability = 0.0
    n = len(document)
    for word in document:
        p_word = estimator.find_probability(word)
        if p_word == 0:
            # if the probability is 0, log p = -inf, so -1/n * sum (...) = inf
            # and exp(inf) = inf, so we return inf
            return math.inf
        log_probability += math.log(p_word)
    return math.exp(-1/ float(n) * log_probability)
