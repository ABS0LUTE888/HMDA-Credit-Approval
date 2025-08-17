# HMDA-Credit-Approval

A simple, interpretable credit approval model built on HMDA data using decision trees

## About

This project explores credit approval modeling using publicly available HMDA (Home Mortgage Disclosure Act) data. The
primary objective is to classify loan applications as approved or denied

### Topics

- AI / Machine Learning
- Credit Analytics
- Decision Tree Models
- Python

## Project Structure

```
HMDA-Credit-Approval/
├─ configs/
│ └─ …                    # (configuration files for training, preprocessing, etc.)
├─ src/
│ ├─ data/
│ │ ├─ download.py        # Script to fetch raw HMDA dataset from source
│ │ ├─ clean.py           # Cleans and preprocesses raw data (handling NaNs, formatting, etc.)
│ │ └─ build_dataset.py   # Builds final dataset ready for training (feature engineering, splits)
│ └─ scripts/
│   ├─ train.py           # Trains decision tree model on prepared dataset and saves model
│   └─ evaluate.py        # Loads a trained model and test dataset, computes evaluation metrics
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ABS0LUTE888/HMDA-Credit-Approval.git
   cd HMDA-Credit-Approval
   ```

2. Create and activate a virtual environment (optional, but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Get raw dataset:
   ```bash
   python -m src.data.download
   ```

2. Clean it up:
   ```bash
   python -m src.data.clean
   ```

3. Build final dataset ready for training:
   ```bash
   python -m src.data.build_dataset
   ```

4. Run the training script:

   ```bash
   python -m src.scripts.train
   ```

5. Run the evaluation script:

   ```bash
   python -m src.scripts.evaluate
   ```

6. _TBD_

## Models

### DecisionTreeClassifier (Scikit-learn)

To train or evaluate with a `Decision Tree`:

```bash
python -m src.scripts.train model=decision_tree
python -m src.scripts.evaluate model=decision_tree
```

Performance:

```json
{
  "auroc": 0.7831595589465814,
  "accuracy": 0.7396051706557055,
  "precision": 0.8683231130002981,
  "recall": 0.7709753220706146,
  "f1": 0.8167587760526401
}
```

### XGBClassifier (XGBoost)

To train or evaluate with `XGBoost`:

```bash
python -m src.scripts.train model=xgboost
python -m src.scripts.evaluate model=xgboost
```

Performance:

```json
{
  "auroc": 0.830525842932744,
  "accuracy": 0.7645003659025991,
  "precision": 0.89519747863669,
  "recall": 0.7782451454194147,
  "f1": 0.8326345724983121
}
```

## Coming Soon

- REST API
- Dockerfile for easy deployment
- Other models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.