# Import Libraries
from keras.datasets import mnist
import numpy as np
import math
np.set_printoptions(linewidth=np.inf)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Data containers for the program
training_data = []
testing_data = []
classification_labels = []

# Number of digits to test, 0-1 for this example
digits = 1

# Diagnostics Table
diagnostics_table = [['Digits', 'Training images:', 'Testing images:']]

# Filter out the data we need from the MNIST dataset
for i in range(digits + 1):
    training_data.append(X_train[np.where(Y_train == i)])
    testing_data.append(X_test[np.where(Y_test == i)])
    classification_labels.append(i)
    diagnostics_table.append([i, training_data[i].shape[0], testing_data[i].shape[0]])

# Print out response of data acquisition
print('Your trainset and testset are generated successfully!\n')
for row in diagnostics_table:
    print("{: >10} {: >20} {: >20}".format(*row))

# Prints out a digit for display purposes
# print(training_data[1][0])

# Extract the following features:
# Feature1:The average brightness of each image (average all pixel brightness values within a whole image array)
# Feature2:The standard deviation of the brightness of each image (standard deviation of all pixel brightness values within a whole image array)
def extract_features(mnist_ndarray):
  # Create a new empty numpy array with 2 columns [[Feature1, Feature2]]
  feature_values = np.empty((0, 2), float)
  # Find the mean and std deviation for each image
  for image in mnist_ndarray:
    feature_values = np.append(feature_values, np.array([[image.mean(), image.std()]]), axis=0)
  return feature_values


# Extract the features for training
training_features = []
for train_set in training_data:
    training_features.append(extract_features(train_set))

# Extract the features for testing
testing_features = []
for test_set in testing_data:
    testing_features.append(extract_features(test_set))


# Class for the GNB Classifier
class GaussNaiveBayesClassifier():
    def __init__(self, training_features, labels):
        self.priors = np.bincount(labels) / len(labels)
        self.means, self.variances = self.__calculate_parameters__(training_features)

    # Calculates the mean and variance of each feature for use in Bayes  Formula
    def __calculate_parameters__(self, training_features):
        means = []
        variances = []
        for data_set in training_features:
            feature_mean = []
            feature_variance = []
            for feature in range(data_set.shape[1]):
                feature_mean.append(data_set[:, feature].mean())
                feature_variance.append(data_set[:, feature].var())
            means.append(feature_mean)
            variances.append(feature_variance)
        return means, variances

    # Gets the product of all values in list
    def __get_products__(self, x):
        product = 0
        for i in x:
            if product == 0:
                product += i
            else:
                product *= i
        return product

    # Calculated the probability
    def get_probability(self, x):
        sum = 0
        probability = []
        total_probabilites = []
        # Loop through each label probability
        for j in range(len(self.priors)):
            dist = []
            # Loop through each feature of the input x
            for i in range(len(x)):
                # Get the value from the formula using the parameters
                dist.append(1 / np.sqrt(2 * np.pi * self.variances[j][i]) * np.exp(
                    -0.5 * ((x[i] - self.means[j][i]) / np.sqrt(self.variances[j][i])) ** 2))
            # Get the product for all in the list, multiplied by the prior label probability
            product = self.__get_products__(dist) * self.priors[j]
            sum += product
            # store this probability value for later (numerator part)
            probability.append(product)
            # find the pobabilites with the sum now (bringing in the denominator part)
        for prob in probability:
            total_probabilites.append(prob / sum)
        return total_probabilites

    # Predicts the label based on the probability
    def predict(self, x):
        # Check the probabilites
        result = self.get_probability(x)
        # Convert the results to a numpy array
        result = np.array(result)
        # Use argmax to find the index of the highest digit
        return result.argmax()

    # Returns the paramters for the classification formula
    def get_parameters(self):
        model_params = {}
        for param in range(len(self.priors)):
            for feature in range(len(self.means[param])):
                model_params[f'Mean of digit {param} feature {feature}'] = self.means[param][feature]
                model_params[f'Variance of digit {param} feature {feature}'] = self.variances[param][feature]
        return model_params

# Tests the accuracy of prediction
def accuracy_test(Classifier, test_features, label):
  result_sum = 0
  for test in test_features:
    result = Classifier.predict(test)
    if result == label:
      result_sum += 1
  return result_sum/len(test_features)


# Create a new classifier
classifier = GaussNaiveBayesClassifier(training_features, classification_labels)

# Get the parameters for classifier
report = classifier.get_parameters()

# Diagnostics Table
diagnostics_table = [['Digits', 'Accuracy']]

for test in range(len(testing_features)):
  accuracy = accuracy_test(classifier, testing_features[test], test)
  report[f'Accuracy for digit {test}'] = accuracy
  diagnostics_table.append([test, "{:.5%}".format(accuracy)])

for row in diagnostics_table:
    print("{: >10} {: >20}".format(*row))

# Generate the report of the model
for (key, value) in report.items():
    print(f'{key: >30}: {value: > 20}')