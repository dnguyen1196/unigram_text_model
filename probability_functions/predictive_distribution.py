

class PredictiveDistribution(object):
    def __init__(self, metadata, prior):
        self.metadata = metadata
        self.prior = prior
        self.dictionary = metadata["dictionary"]
        self.N = metadata["N"]
        self.K = metadata["K"]

    def find_probability(self, word):
        if word in self.dictionary:
            mk = self.dictionary[word]
        else:
            mk = 0
        ak = self.prior[word]
        a0 = self.prior["a0"]
        return float(mk + ak)/(self.N + a0)
