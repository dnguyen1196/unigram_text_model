import math
"""
DocumentLoader
This class loads the training and test text files to build the dictionary
For the train file, it counts the number of times each word appears
For the test file, it adds word that are not covered in the training text file
This creates a vocabulary for the entire train + test text
"""


class DocumentExtractor(object):
    def __init__(self, train_file, test_file, training_size=math.inf, min_word=None):
        self.train_file = train_file
        self.test_file = test_file
        self.training_size = training_size
        self.metadata = {}
        self.min_word = min_word
        self.total_word_count = 0
        self.load_text_data()

    def load_text_data(self):
        dictionary = {}

        # Load training data
        train_data = open(self.train_file, "r")
        for line in train_data:
            words = line.strip().split()
            for word in words:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

                # Increase word count
                self.total_word_count += 1
                if self.total_word_count >= self.training_size:
                        break
            if self.total_word_count >= self.training_size:
                break
        train_data.close()

        for test_file in self.test_file:
            test_data = open(test_file, "r")
            for line in test_data:
                words = line.strip().split()
                for word in words:
                    if word not in dictionary:
                        dictionary[word] = 0
            test_data.close()

        # Remove all words that appears less than some min_word threshold
        if self.min_word is not None:
            for word in dictionary.keys():
                if 0 < dictionary[word] < self.min_word:
                    count = dictionary[word]
                    dictionary[word] = 0
                    self.total_word_count -= count

        self.metadata["K"] = len(dictionary)
        self.metadata["N"] = self.total_word_count
        self.metadata["dictionary"] = dictionary

    def get_metadata(self):
        return self.metadata
