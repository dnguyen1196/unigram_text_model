import math
"""
DocumentLoader
This class loads the training and test text files to build the dictionary
For the train file, it counts the number of times each word appears
For the test file, it adds word that are not covered in the training text file
This creates a vocabulary for the entire train + test text
"""
class DocumentLoader(object):
    def __init__(self, train_file, test_file, training_size=math.inf):
        self.train_file = train_file
        self.test_file = test_file
        self.training_size = training_size
        self.metadata = {}
        self.load_text_data()

    def load_text_data(self):
        dictionary = {}
        total_word_count = 0

        train_data = open(self.train_file)
        for line in train_data:
            words = str(line).rstrip().split(' ')
            for word in words:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 0
                total_word_count += 1
                if total_word_count > self.training_size:
                    break
            if total_word_count > self.training_size:
                break

        train_data.close()

        for test_file in self.test_file:
            test_data = open(test_file, "r")
            for line in test_data:
                words = str(line).rstrip().split(" ")
                for word in words:
                    if word not in dictionary:
                        dictionary[word] = 0
            test_data.close()

        self.metadata["K"] = len(dictionary)
        self.metadata["N"] = total_word_count
        self.metadata["dictionary"] = dictionary

    def get_metadata(self):
        return self.metadata