# src/models/train_model.py
# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data transformation scripts to turn raw data from (input_filepath) into
    transformed data ready to be modeled (saved in output_filepath).
    """
    logger = logging.getLogger(__name__)
    logger.info('Transforming data')

    # Load data
    data = pd.read_csv(input_filepath)

    # Load imputer and scaler from pkl files
    imputer = joblib.load('../models/simple_imputer.pkl')
    scaler = joblib.load('../models/min-max-scaler.pkl')

    # Transform data
    transformed_data = transform_data(data, imputer, scaler)

    # Formatting data
    X, customer_ids = formatting(transformed_data)

    # Save the processed data
    X.to_csv(output_filepath, index=False)
    logger.info(f'Data saved to {output_filepath}')

def transform_data(df, imputer, scaler):
    logger = logging.getLogger(__name__)
    logger.info('Applying transformations')
    """
    Function to perform necessary data transformations like encoding and scaling.
    """
    # Encoding categorical variables and scaling
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})
    df['owns_car'] = df['owns_car'].map({'N': 0, 'Y': 1})
    df['owns_house'] = df['owns_house'].map({'N': 0, 'Y': 1})

    # One-hot encoding for 'occupation_type'
    df = pd.get_dummies(df, columns=['occupation_type'], prefix=['ot'])
    # imputting missing values
    df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    # Scaling the dataset
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df

def formatting(df):
    logger = logging.getLogger(__name__)
    logger.info('Formatting data')
    # 'customer_id' is not a feature used in the model, 
    # but an identifier to keep for later reference.
    if 'customer_id' in df.columns:
        customer_ids = df['customer_id']
        X = df.drop(columns=['customer_id'], axis=1)
    else:
        customer_ids = None
        X = df
    
    return X, customer_ids

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
