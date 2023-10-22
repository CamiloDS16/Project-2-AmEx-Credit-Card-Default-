# src/data/make_dataset.py
# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from dotenv import find_dotenv, load_dotenv

def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

def clean_data(df):
    # renaming columns
    df = df.rename(columns={'credit_limit_used(%)': 'credit_limit_used_pctg'})
    # Filling missing values for 'no_of_children', 'owns_car', 'migrant_worker', 'total_family_members' with mode
    for col in ['no_of_children', 'owns_car', 'migrant_worker', 'total_family_members']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Handling 'XNA' values in 'gender'
    df['gender'] = df['gender'].replace('XNA', df['gender'].mode()[0])

    return df

def feature_engineering(df):
    # dropping columns to mitigate multicollinearity
    df = df.drop(columns=['name', 'credit_limit', 'no_of_children'], axis=1)
    # combining 'prev_defaults' and 'default_in_last_6months' into 'total_defaults'
    if 'prev_defaults' in df.columns and 'default_in_last_6months' in df.columns:
        df['total_defaults'] = df['prev_defaults'] + df['default_in_last_6months']
        df = df.drop(columns=['prev_defaults', 'default_in_last_6months'])
    
    return df

def transform_data(df, imputer, scaler):
    # Encoding categorical variables and scaling
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})
    df['owns_car'] = df['owns_car'].map({'N': 0, 'Y': 1})
    df['owns_house'] = df['owns_house'].map({'N': 0, 'Y': 1})

    # One-hot encoding for 'occupation_type'
    df = pd.get_dummies(df, columns=['occupation_type'], prefix=['ot'])
    # imputing missing values
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    # Scaling the dataset
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df

def formatting(df):
    # customer_id' is not a feature used in the model, 
    # but an identifier to keep for later reference.
    if 'customer_id' in df.columns:
        customer_ids = df['customer_id']
        X = df.drop(columns=['customer_id'], axis=1)
    else:
        customer_ids = None
        X = df
    
    return X, customer_ids

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data = load_data(input_filepath)

    if data is not None:
        cleaned_data = clean_data(data)
        engineered_data = feature_engineering(cleaned_data)
        engineered_data.to_csv(output_filepath, index=False)
        logger.info(f'Data saved to {output_filepath}')
    else:
        logger.error('No data to save')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
