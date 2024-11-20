# LassoGRU: A Framework for Feature Selection and Prediction Using LassoCV and GRU

## Overview
**LassoGRU** is a Python-based tool that integrates LassoCV for feature selection and a GRU-based neural network for regression.

## Key Features
1. **Feature Selection**: Leverages LassoCV to select informative features while removing irrelevant ones.
2. **Regression**: Uses a GRU-based neural network to achieve high predictive accuracy.
3. **Reusable Outputs**: Saves feature selectors, scalers, and trained models for easy application on new datasets.

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your_username/lassoGRU.git
cd lassoGRU
```

## Install Dependencies
```bash
pip install -r requirements.txt
# or
conda create --name lassoGRU_env --file requirements.txt
conda activate lassoGRU_env
```

## Usage 
### Command-Line Interface
LassoGRU is designed as a command-line tool. To use it, run the following command:
```bash
python lassoGRU.py --input_csv <path_to_your_csv> --label_column <label_column_name> --output_model_path <output_prefix> --validation_split <validation_portion>
```

#### Arguments: 
| Argument            | Type   | Default  | Description                                                                 |
|---------------------|--------|----------|-----------------------------------------------------------------------------|
| `--input_csv`       | string | Required | Path to the input CSV file with features and labels.                        |
| `--label_column`    | string | Required | Name of the column containing the target labels.                            |
| `--output_model_path`| string | Required | Path to save the trained model and preprocessing files.                     |
| `--validation_split`| float  | 0.2      | Fraction of the dataset to use for validation (e.g., `0.2` means 20%).      |

### Outputs The script will generate the following files: 
1. **`<output_model_path>.h5`**: The trained GRU model. 
2. **`<output_model_path>_scaler.pkl`**: The MinMaxScaler for feature scaling. 
3. **`<output_model_path>_selector.pkl`**: The LassoCV-based feature selector.

