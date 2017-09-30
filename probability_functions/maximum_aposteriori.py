

class MaximumAPosteriori(object):
    def __init__(self, metadata, prior):
        self.metadata = metadata
        self.prior = prior
        self.dictionary = metadata["dictionary"]
        self.N = metadata["N"]
        self.K = metadata["K"]

    """
    Returns the probability(word | model)
    """
    def find_probability(self, word):
        if word in self.dictionary:
            m_k = self.dictionary[word]
        else:
            m_k = 0
        a_k = self.prior[word]
        a_0 = self.prior["a0"]
        return float(m_k + a_k - 1)/(self.N + a_0 - self.K)
