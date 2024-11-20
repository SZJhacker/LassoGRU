#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Define LassoGRU framework
def create_model():
    model = Sequential()
    model.add(GRU(units=64, return_sequences=True))
    model.add(GRU(units=32))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer='RMSprop', loss='mae', metrics=['mae'])
    return model

# Reshape features for GRU input
def reshape_for_gru(data):
    return np.reshape(data, (data.shape[0], 1, -1)).astype(np.float32)

# Main function for feature selection and model training
def train_lasso_gru(input_csv, label_column, output_model_path):
    # Load and preprocess data
    data = pd.read_csv(input_csv)
    labels = data[label_column].values
    features = data.drop(columns=[label_column]).values

    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # Lasso feature selection
    lasso = LassoCV(cv=5, n_jobs=-1)
    lasso.fit(features_scaled, labels)
    selector = SelectFromModel(estimator=lasso, prefit=True)
    features_selected = selector.transform(features_scaled)

    # Save scaler and selector for reuse
    joblib.dump(scaler, f"{output_model_path}_scaler.pkl")
    joblib.dump(selector, f"{output_model_path}_selector.pkl")

    # Prepare data for GRU
    features_gru = reshape_for_gru(features_selected)

    # Create and train GRU model
    model = create_model()
    callback = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(features_gru, labels, epochs=100, validation_split=validation_split, callbacks=[callback])

    # Save trained model
    model.save(f"{output_model_path}.h5")
    print(f"Model saved to {output_model_path}.h5")

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="LassoGRU: A framework for feature selection and regression")
    parser.add_argument("--input_csv", type=str, required=True, 
                        help="Path to the input CSV file (must contain features and one label column).")
    parser.add_argument("--label_column", type=str, required=True, 
                        help="Name of the label column in the input CSV file.")
    parser.add_argument("--output_model_path", type=str, required=True, 
                        help="Path prefix for saving the trained model and preprocessing tools.")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Fraction of data to use for validation (default: 0.2).")
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_model_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Train the model
    train_lasso_gru(input_csv=args.input_csv, 
                    label_column=args.label_column, 
                    output_model_path=args.output_model_path,
                    validation_split=args.validation_split)

if __name__ == "__main__":
    main()
