# %%
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import logging
import sys

def load_data(path):
    """
    Load the data and drop Unnamed column.
    
    Parameters:
    path where the data is stored.
    
    Returns:
    Dataframe.
    """
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# load pkl files for model, imputer, scaler  
def load_assets(model_path, imputer_path, scaler_path):
    """
    Load the model, imputer, and scaler from disk.
    """
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    return model, imputer, scaler

def clean_data(df):
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

def transform_data(df, imputer, scaler, original_features):
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
    # Assuming 'customer_id' is not a feature used in the model, 
    # but an identifier that you'll want to keep for later reference.
    if 'customer_id' in df.columns:
        customer_ids = df['customer_id']
        X = df.drop(columns=['customer_id'], axis=1)
    else:
        customer_ids = None
        X = df
    
    return X, customer_ids

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

# predictions

def make_predictions(X, model, customer_ids):
    """
    Make predictions, and return results.
    """
    predictions = model.predict(X)
    results = pd.DataFrame({'customer_id': customer_ids, 'prediction_default': predictions})
    return results

def main():
    # paths
    data_path = '../data/raw/test.csv'
    imputer_path = '../model-xgb/simple_imputer.pkl'
    scaler_path = '../model-xgb/min-max-scaler.pkl'
    xgb_model_path = '../model-xgb/best_xgb_model.pkl'
    original_features = ['age', 'gender', 'owns_car', 'owns_house', 'net_yearly_income',
           'no_of_days_employed', 'occupation_type', 'total_family_members',
           'migrant_worker', 'yearly_debt_payments', 'credit_limit_used_pctg',
           'credit_score', 'total_defaults']

    logger.info("Starting the data processing pipeline")

    # Load data
    logger.info("Loading data")
    df = load_data(data_path)
    if df is None:
        logger.error(f"Failed to load data from {data_path}")
        sys.exit(1)

    # Load assets
    try:
        logger.info("Loading model, imputer, and scaler")
        model, imputer, scaler = load_assets(xgb_model_path, imputer_path, scaler_path)
    except FileNotFoundError as e:
        logger.error(f"Asset not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while loading assets: {e}")
        sys.exit(1)

    # Data cleaning
    logger.info("Cleaning data")
    cleaned_data = clean_data(df)

    # Feature engineering
    logger.info("Performing feature engineering")
    engineered_data = feature_engineering(cleaned_data)

    # Formatting data
    logger.info("Formatting data for predictions")
    X, customer_ids = formatting(engineered_data)

    # Data transformation
    logger.info("Transforming data")
    X = transform_data(X, imputer, scaler, original_features)

    # Preparing data for prediction
    logger.info("Preparing data for prediction")
    X = prepare_for_prediction(X, model)

    # Making predictions
    logger.info("Making predictions")
    results = make_predictions(X, model, customer_ids)

    # Save results
    try:
        output_path = '../reports/documentation/results.csv'
        results.to_csv(output_path, index=False)
        logger.info(f"Results successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    try:
        main()
    except Exception as e:
        logger.exception("Fatal error in main loop")
        sys.exit(1)