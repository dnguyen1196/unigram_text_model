"""
Maximum likelihood estimator
Find the probability of a word appearing in a document based on
information obtained from training model
"""
class MaxLikelihoodEstimator(object):
    def __init__(self, metadata):
        self.metadata = metadata
        self.N = metadata["N"]

    """
    Returns the probability(word | model)
    """
    def find_probability(self, word):
        if word not in self.metadata["dictionary"]:
            return 0
        return float((self.metadata["dictionary"])[word])/(self.N)