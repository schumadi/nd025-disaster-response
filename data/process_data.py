"""
Module ETL pipeline
"""

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories from the respective paths.
    The categories are attributed to the messages.
    The result is returned as a DataFrame

    Arguments:
        messages_filepath -- Path to the csv file containing the messages
        categories_filepath -- Path to the csv file containing the categories

    Returns:
        DataFrame with messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')

    return df


def clean_data(df):
    """
    Transforms the category column into one column for each category
    Duplicate rows and NaN-messages are be dropped. 
    Rows with related value not 0 or 1 are dropped.
    The id, genre and original column are dropped.

    Arguments:
        df -- The dataframe to be cleaned

    Returns:
        Cleaned dataframe
    """
    # create a dataframe consisting of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # The rows contain values like "medical_products-0"
    # Take the first row and set the column names of the categories dataframe
    # to the part in front of the dash
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Drop the original categories column
    df.drop('categories', inplace=True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates; rows with related not 0 or 1; NaN messages; the id and original column
    # Drop child_alone because it's always 0 and the model cannot learn from that
    df.drop_duplicates(inplace=True)
    df.drop(df[(df.related != 0) & (df.related != 1)].index, axis=0, inplace=True)
    df.dropna(subset=['message'], axis=0, inplace=True)
    df.drop(['id', 'original'], axis=1, inplace=True)
    df.drop('child_alone', axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Loads the contents of the dataframe df to sqlite db

    Arguments:
        df -- The dataframe to be stored
        database_filename -- The filepath of the sqlite db
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    """
    * Loads the messages and categories datasets
    * Merges the two datasets
    * Cleans the data
    * Stores it in a SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
