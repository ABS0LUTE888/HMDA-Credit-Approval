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

   _Note_: This project uses Hydra for configuration management, so you can override any setting straight from the CLI
   without
   editing YAMLs. For example:
   ```bash
   python -m src.data.download year=2023 
   
   # or
   
   python -m src.scripts.train model=xgboost seed=67
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
  "auroc": 0.7861465196838779,
  "accuracy": 0.7097934651120077,
  "precision": 0.8881341616971602,
  "recall": 0.7030027797283301,
  "f1": 0.7847983013774197
}
```

Confusion Matrix:

<img width="960" height="720" alt="confusion_matrix" src="https://github.com/user-attachments/assets/a1423d03-ea06-48c7-aa19-58eb3c0d9160" />

ROC Curve:

<img width="960" height="720" alt="roc_curve" src="https://github.com/user-attachments/assets/cd773881-c0be-4dd3-941f-8b06d9fd826d" />

### XGBClassifier (XGBoost)

To train or evaluate with `XGBoost`:

```bash
python -m src.scripts.train model=xgboost
python -m src.scripts.evaluate model=xgboost
```

Performance:

```json
{
  "auroc": 0.8111004309136769,
  "accuracy": 0.7393316793345928,
  "precision": 0.8922882998344719,
  "recall": 0.7434411721060304,
  "f1": 0.8110923852200005
}
```

Confusion Matrix:

<img width="960" height="720" alt="confusion_matrix" src="https://github.com/user-attachments/assets/4a8c863d-b079-4bbc-8d9c-b359df3898ab" />

ROC Curve:

<img width="960" height="720" alt="roc_curve" src="https://github.com/user-attachments/assets/780acc48-c341-4d89-a079-7fb7d7e6b132" />

### Adding custom models

You can easily add your own models by following these steps:

1. Create the model file at `src/models/your_model.py` and implement the model class.
   ```python
   # src/models/your_model.py   
   from src.models.base import BaseModel, DataBundle # change to .base
   from src.models.registry import register # change to .registry
   
   @register("your_model") # must match the YAML name
   class YourModel(BaseModel):
      def __init__(self, cfg):
        super().__init__(cfg)
        ...
      
      def train(self, data: DataBundle):
        ...

      def predict(self, X):
        ...

      def predict_proba(self, X):
        ...

      def evaluate(self, data: DataBundle):
        ...

      def export_pipeline(self, preprocessor_path, out_path):
        ...
   ```
   Key points:
    - Inherit from either `src.models.base.BaseModel` or
      `src.models.base.SciKitModel` (which already features most of the methods above).
    - The decorator `@register("your_model")` must use the same string that appears in your YAML config (next step).
2. Create the config at `configs/models/your_model.yaml`:
   ```yaml
   name: your_model
   params: {} # Fixed params that are passed to model instance
   tuning: # Training params
      n_jobs: 4
      cv: 5
      scoring: roc_auc
      verbose: 4
      param_grid: # Params passed to GridSearchCV
        max_depth: [ 4, 6, 8, 10 ]
   ```
3. Train and evaluate your model:
   ```bash
   python -m src.scripts.train model=your_model
   python -m src.scripts.evaluate model=your_model
   ```

## Coming Soon

- REST API
- Dockerfile for easy deployment
- Other models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
