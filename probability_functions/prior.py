
class Prior(object):
    def __init__(self, metadata, alpha_prime):
        self.dictionary = metadata["dictionary"]
        self.alpha_prime = alpha_prime
        self.prior = {}

    def get_prior(self):
        K = len(self.dictionary)
        alpha = [self.alpha_prime for i in range(K)] # TODO: what to do with alpha, how to initialize

        for word in self.dictionary.keys():
            self.prior[word] = self.alpha_prime

        self.prior["a0"] = self.alpha_prime * K # TODO: how to automatically update this

        return self.prior