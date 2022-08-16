'''
Module ML pipeline
'''
#pylint: disable=wrong-import-position

import sys
import warnings

warnings.filterwarnings('ignore')

import pickle
import re

import nltk
import numpy as np
import pandas as pd

nltk.download('popular')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from lightgbm import LGBMClassifier
from optuna.distributions import IntUniformDistribution
from optuna.integration import OptunaSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    The messages stored in an sqlite database are loaded to a dataframe.
    The dataframe is split into a feature matrix an labels

    Arguments:
        database_filepath -- Path to the sqlite database containing the messages

    Returns:
        Feature matrix X, labels Y and the categories
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)

    X = df['message']
    Y = df.drop(columns=['message', 'genre'])

    return X, Y, Y.columns


def tokenize(text):
    """
    Cleans, tokenizes and lemmatizes the input
    Punctuation, URLs and english stopwords are removed
    Text converted to lowercase

    Arguments:
        text -- Text to be prepared

    Returns:
        Lemmata of the cleaned tokens of the text
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Replace URLs by the string "URL"
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "URL")

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return lemmatized


def build_model():
    """
    Creates the model to be used to label the messages.
    CountVectorizer and TfidfTransformer are integrated in the pipeline

    Returns:
        Best LGBMClassifier model found by OptunaSearchCV
    """

    # Define a pipeline, parameter distributions for optuna and search for the best model
    pipeline = Pipeline([
                                ('vect', CountVectorizer(tokenizer = tokenize)),
                                ('tfidf', TfidfTransformer()),
                                ('clf', MultiOutputClassifier(LGBMClassifier(random_state=42, objective='binary', n_jobs=-1)))
                            ])
    optuna_param_dist = {
                        'clf__estimator__num_leaves': IntUniformDistribution(20,40),
                        'clf__estimator__max_depth': IntUniformDistribution(3,15),
                        'clf__estimator__min_data_in_leaf': IntUniformDistribution(10,100)
                    }           
    clf = OptunaSearchCV(pipeline, optuna_param_dist, cv=5, random_state=42, verbose=2, n_jobs=-1)


    return clf



def evaluate_model(model, X_test, Y_test, category_names):
    """
    The predictions for the test set are evaluted.
    Precision, Recall, F1-Score, Support and Averages are computed and returned as a dataframe

    Arguments:
        model -- the fitted model to be used for predictions
        X_test -- The test set
        y_test -- The lables for the tests set
        category_names -- The categories a message can be associated with

    """
    Y_hat = model.predict(X_test)

    row_list = []
    for col_number, col_name in enumerate(category_names):
        for label in np.unique(Y_hat.ravel()):
            # Get the precision, reacall, f_score and support for column number col_number
            pre_rec_f1_supp = precision_recall_fscore_support(  y_true = Y_test.iloc[:,col_number],
                                                                y_pred = Y_hat[:,col_number]
                                                                )

            # Collect the results.
            # The dicts are collected in row_list which is converted to a Dataframe at the end.
            result = { 'Label': col_name, 'Value': label, 'Precision': pre_rec_f1_supp[0][label],
                        'Recall': pre_rec_f1_supp[1][label], 'F-1 Score': pre_rec_f1_supp[2][label],
                        'Support': pre_rec_f1_supp[3][label]
                        }
            row_list.append(result)
    # Add averages to the result
    for avg in ['micro', 'macro', 'weighted']:
        average = precision_recall_fscore_support(y_true = Y_test, y_pred = Y_hat, average=avg)
        result = {
                    'Label': 'Average', 'Value': avg, 'Precision': average[0], 'Recall': average[1],
                    'F-1 Score': average[2], 'Support': '-'
                    }
        row_list.append(result)

    res_df=pd.DataFrame(row_list)
    res_df.set_index(['Label', 'Value'], inplace=True)
    print(res_df)



def save_model(model, model_filepath):
    """
    The model is saved as a pickle file.

    Arguments:
        model -- the final model
        model_filepath -- Filepath of the pickle file to save the model
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    * Loads data from the SQLite database
    * Splits the dataset into training and test sets
    * Builds a text processing and machine learning pipeline
    * Trains and tunes a model using GridSearchCV
    * Outputs results on the test set
    * Exports the final model as a pickle file
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
