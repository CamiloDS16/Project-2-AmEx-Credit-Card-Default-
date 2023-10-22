# src/__init__.py
from .data.make_dataset import load_data
from .features.build_features import clean_data, feature_engineering
from .models.train_model import transform_data, formatting  # adjust to predict_model if necessary

# Add the following lines to enable logging in this module
import logging

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
