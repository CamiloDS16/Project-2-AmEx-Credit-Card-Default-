# src/features/build_features.py
# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (input_filepath) into
        cleaned and engineered data ready to be analyzed (saved in output_filepath).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # Load data
    data = pd.read_csv(input_filepath)

    # Clean data
    cleaned_data = clean_data(data)
    
    # Feature Engineering
    engineered_data = feature_engineering(cleaned_data)
    
    # Save the processed data
    engineered_data.to_csv(output_filepath, index=False)
    logger.info(f'Data saved to {output_filepath}')


def clean_data(df):
    logger = logging.getLogger(__name__)
    logger.info('Cleaning data')
    """
    Function to clean data by handling missing values and anomalies.
    """
    # renaming columns
    df = df.rename(columns={'credit_limit_used(%)': 'credit_limit_used_pctg'})
    # Filling missing values for 'no_of_children', 'owns_car', 'migrant_worker', 'total_family_members' with mode
    for col in ['no_of_children', 'owns_car', 'migrant_worker', 'total_family_members']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Handling 'XNA' values in 'gender'
    df['gender'] = df['gender'].replace('XNA', df['gender'].mode()[0])

    # Your data might have specific anomalies that you discovered during EDA.
    # Include code to handle those anomalies here.
    
    return df


def feature_engineering(df):
    logger = logging.getLogger(__name__)
    logger.info('Engineering features')
    """
    Function for feature engineering, like combining certain features for new insights.
    """
    # dropping columns to mitigate multicollinearity
    df = df.drop(columns=['name', 'credit_limit', 'no_of_children'], axis=1)
    # combining 'prev_defaults' and 'default_in_last_6months' into 'total_defaults'
    if 'prev_defaults' in df.columns and 'default_in_last_6months' in df.columns:
        df['total_defaults'] = df['prev_defaults'] + df['default_in_last_6months']
        df = df.drop(columns=['prev_defaults', 'default_in_last_6months'])
    
    return df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
