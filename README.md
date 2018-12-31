# Sentiment-Analyis-of-IMDB-dataset

# Overview
I have used the Internet Movie Database (IMDb) movie reviews dataset to illustrate the workflow. This dataset contains movie reviews posted by people on the IMDb website, as well as the corresponding labels (“positive” or “negative”) indicating whether the reviewer liked the movie or not. This is a classic example of a sentiment analysis problem.

Link to the dataset: http://ai.stanford.edu/~amaas/data/sentiment/

# Dataset 
The core dataset contains 50,000 reviews split evenly into 25k trainand 25k test sets. The overall distribution of labels is balanced (25k pos and 25k neg). We also include an additional 50,000 unlabeled documents for unsupervised learning.

# Explorin the Data
For choosing the model I have calculated the median for the training dataset(25000 samples) i.e., 174.0 and thebelow ratio 
      ratio = number of samples/number of words per sample ratio
      ratio = 25000/median
I have followed the below rule as suggested in Google Text classification documentation.
1.If this ratio is less than 1500, tokenize the text as n-grams and use a simple multi-layer perceptron (MLP) model to classify them.
  a. Split the samples into word n-grams; convert the n-grams into vectors.
  b. Score the importance of the vectors and then select the top 20K using the scores.
  c. Build an MLP model.
2.If the ratio is greater than 1500, tokenize the text as sequences and use a sepCNN model to classify them.
  a. Split the samples into words; select the top 20K words based on their frequency.
  b. Convert the samples into word sequence vectors.
I have implemented Tf-idf encoding beacuse tf-idf encoding is marginally better than Count encoding and One-hot encoding in terms of accuracy (on average: 0.25-15% higher), and using this method for vectorizing n-grams. However it occupies more memory (as it uses floating-point representation) and takes more time to compute, especially for large datasets (can take twice as long in some cases).
More importantly,the accuracy peaks at around 20,000 features for many datasets Adding more features over this threshold contributes very little and sometimes even leads to overfitting and degrades performance.
[source: Google Text classification]

# Model Implementation
This multi-layer perceptrons (MLPs) are implemented using Keras.
Models that process the tokens independently (not taking into account word order) as n-gram models. Simple multi-layer perceptrons (including logistic regression), gradient boosting machines and support vector machines models all fall under this category; they cannot leverage any information about text ordering.After comparing the performance of some of the n-gram models mentioned above and observed that multi-layer perceptrons (MLPs) typically perform better than other options. MLPs are simple to define and understand, provide good accuracy, and require relatively little computation.

# Evaluting model performance
Implemented the both MLP model and Random ForestClassifier  model and achieved the better accuracy with Multi-Layer Perceptrons(ANN).
Multi-Layer Perceptrons(ANN) accuracy:
accuracy on test set : 0.88784 
that is 22196 are predicted correct out of 25000

Random Forest Classifier accuracy:
accuracy on test set : 0.7664
that is 19160 are predicted correct out of 25000
