# CS626_CRF_POS_Tagger

This project implements a Conditional Random Field (CRF) based Part-of-Speech (POS) tagger using both LSTM and GRU models.

## Prerequisites

- Python 3.10.11

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/imranadas/CS626_CRF_POS_Tagger
   cd CS626_HMM_CRF_Tagger
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv py_venv
   source py_venv/bin/activate  # On Windows, use `py_venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r Requirements.txt
   ```

## Usage

### Training

To train the models, run:

```
python crf_training_LSTM.py
python crf_training_GRU.py
```

This will train both LSTM and GRU models and save them in the `Models` directory.

### Evaluation

To evaluate the trained models, run:

```
python crf_evaluation_LSTM.py
python crf_evaluation_GRU.py
```

This will generate evaluation metrics for both models and save them in the `Models_Metrics` directory.

### Inference

You can use the trained models for inference in two ways:

1. Console-based inference:
   ```
   python crf_inference.py
   ```

2. Streamlit UI:
   ```
   streamlit run crf_app.py
   ```

## Project Structure

- `crf_training_LSTM.py` / `crf_training_GRU.py`: Scripts for training LSTM and GRU models
- `crf_evaluation_LSTM.py` / `crf_evaluation_GRU.py`: Scripts for evaluating LSTM and GRU models
- `crf_inference.py`: Script for console-based inference
- `crf_app.py`: Streamlit app for UI-based inference
- `Models/`: Directory containing trained models
- `Models_Metrics/`: Directory containing evaluation metrics
- `Logs/`: Directory containing log files
- `Model_Config/`: Directory containing model configurations
