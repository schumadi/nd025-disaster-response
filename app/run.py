import json
import plotly
import pandas as pd
import re

import nltk
nltk.download('popular')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

import joblib
from sqlalchemy import create_engine

from optuna.visualization import plot_slice
from optuna.visualization import plot_parallel_coordinate

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Startpage showing the Distribution of Message Genres and Distribution of Categories

    Returns:
        Rendered template master.html
    """
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_df = df.drop(columns=['message', 'genre'])
    category_counts = category_df.mean()*100
    category_names = category_df.columns
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Related Messages in %"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Page to predict labels for messages.

    Returns:
        Template go.html
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# web page that handles user query and displays model results
@app.route('/model')
    """
    Page showing two plots of the optuna hyperparameter optimization.
    
    Returns:
        Template model.html
    """
def show_parameters():

    graphs = [
                {
                    'data': plot_slice(model.study_),

                    'layout': {
                        'title': 'Sliceplot',
                        'yaxis': {
                            'title': "y"
                        },
                        'xaxis': {
                            'title': "x"
                        }
                    }
                },
                {
                    'data': plot_parallel_coordinate(model.study_),

                    'layout': {
                        'title': 'Parallel Relationships',
                        'yaxis': {
                            'title': "y"
                        },
                        'xaxis': {
                            'title': "x"
                        }
                    }
                }
                ]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('model.html', best_trial=model.study_.best_trial, ids=ids, graphJSON=graphJSON)

 

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()