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
│   └─ train.py           # Trains decision tree model on prepared dataset and saves model
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

5. _TBD_

## Coming Soon

- Evaluation script
- XGBoost model
- REST API
- Dockerfile for easy deployment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.