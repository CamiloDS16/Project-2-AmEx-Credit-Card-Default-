# src/models/predict_model.py
# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import pickle
from pathlib import Path

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, model_filepath, output_filepath):
    """ Loads data from the input file, loads the trained model from the model file,
        makes predictions on the data, and saves the results to the output file.
    """
    logger = logging.getLogger(__name__)
    logger.info('Making predictions with the model')

    # Load data
    data = pd.read_csv(input_filepath)
    customer_ids = data['customer_id']  # Assuming customer_id is a column in your data

    # Load model
    with open(model_filepath, 'rb') as file:
        model = pickle.load(file)

    # Prepare data for prediction
    X = prepare_for_prediction(data, model)

    # Make predictions
    results = make_predictions(X, model, customer_ids)

    # Save results to output file
    results.to_csv(output_filepath, index=False)
    logger.info(f'Predictions saved to {output_filepath}')

def prepare_for_prediction(X, model):
    """
    Ensure the order of columns in the new data (X) matches the features used for training the model.
    If there's a mismatch, reorder columns in X to match the training features.
    """
    expected_features = model.get_booster().feature_names
    received_features = list(X.columns)

    # Check if all expected features are present
    if not set(expected_features).issubset(received_features):
        missing_features = set(expected_features) - set(received_features)
        raise ValueError(f"Missing features: {missing_features}")

    # If features are present but in different order, reorder them
    if expected_features != received_features:
        X = X[expected_features]
    
    return X

def make_predictions(X, model, customer_ids):
    """
    Make predictions, and return results.
    """
    predictions = model.predict(X)
    results = pd.DataFrame({'customer_id': customer_ids, 'prediction_default': predictions})
    return results

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
