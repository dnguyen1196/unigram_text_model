#
#   The class prior
#
class Prior(object):
    def __init__(self, metadata, alpha_prime):
        self.dictionary = metadata["dictionary"]
        self.alpha_prime = alpha_prime
        self.prior = {}

    def get_prior(self):
        K = len(self.dictionary)
        for word in self.dictionary.keys():
            self.prior[word] = self.alpha_prime

        self.prior["a0"] = self.alpha_prime * K
        return self.prior
