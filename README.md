## Task 1: model training/prediction and evaluation
Find the perplexity for ML, MAP and predictive distribution function

PP = exp(1/N sum (ln p(w))) for all words

Different distribution function will have a different p(data) function
These functions could be found in the additional note of the programming assignment sheet

The variables that ML, MAP and PD need are mi (the count of the ith word), 
N (total number of words in the training example)
, alphai (the parameter for the ith class in the dirichlet prior), 
K is the total number of distince words in the VOCABULARY


ML -> just need metadata
MAP -> Need prior 
predictive distribution -> also need prior

evidence function -> obviously requires prior


find_perplexity -> should be a function that calls p(word) from the estimators

estimators relies on (metadata, prior)



Questions:
- How to handle huge gamma value (sed)



Requirements
Task 1
- calculate perplexity on test data set
- train 3 different estimators on the initial segment o
of the data (N/128 -> N)
- report the perplexities on the train set + test set
Answer questions: 



Task 2 Model selection
Vary alpha prime for the same training set of 
size N/128
Compite perplexities on the test set
Plot the LOG EVIDENCE and test set perplexity
as function of alpha prime


Task 3 author identification
Train model on some train dataset with
alpha prime = 2, and evaluate perplexit on each of 
the other two texts

How to classify?? ==> like it's just different? Or small perplexity means its 
the same author?

Repeat the above but remove all words <= 50 times



NOTE:
- how to hand le the case where ln(0) = -inf