## Spam Email Classifier
This is a machine learning model built in Python that can predict whether an email is spam or not. The model uses logistic regression algorithm to classify the emails based on their content.

## Dataset
The dataset used in this project is taken from Kaggle, which contains a collection of spam and ham (non-spam) emails. The dataset has been preprocessed and cleaned before training the model.

## Model Training
The logistic regression algorithm is used for training the model. The text data is first transformed into feature vectors using the TfidfVectorizer. Then, the model is trained on the feature vectors and their corresponding labels.

## Model Evaluation
The accuracy of the trained model is evaluated on both the training and testing datasets. The accuracy score for training data is calculated to be around 95%, while for testing data, it is around 94%.

## Prediction
Finally, the user can input a message, and the model will predict whether the message is spam or not. The input text is first transformed into feature vectors using the TfidfVectorizer, and then the logistic regression model makes the prediction.

## Usage
To use this model, you can clone this repository and run the spam_classifier.py file. The user will be prompted to enter a message that needs to be classified as spam or ham.

## Requirements
Python 3.6 or above
pandas
scikit-learn
## References
<a herf src="https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset">Kaggle Spam Dataset</a>