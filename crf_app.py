import os
import json
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from crf_inference import infer_sentence
from crf_training_GRU import CRFModel as GRUModel, Config as GRUConfig
from crf_training_LSTM import CRFModel as LSTMModel, Config as LSTMConfig

def setup_logger(name, log_file, level=logging.INFO):
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler to write logs to the specified file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Get or create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate logging by clearing existing handlers (if needed)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger('crf_app', 'Logs/crf_app_log.log')

# Load model configurations and mappings
def load_model_config(model_type):
    logger.info(f"Loading {model_type} model configuration")
    with open(f'Model_Config/{model_type}_model_params.json', 'r') as f:
        model_params = json.load(f)
    with open(f'Model_Config/{model_type}_word_to_ix.json', 'r') as f:
        word_to_ix = json.load(f)
    with open(f'Model_Config/{model_type}_tag_to_ix.json', 'r') as f:
        tag_to_ix = json.load(f)
    logger.info(f"{model_type} model configuration loaded successfully")
    return model_params, word_to_ix, tag_to_ix

# Initialize and load the selected model
def load_model(model_type):
    logger.info(f"Loading {model_type} model")
    model_params, word_to_ix, tag_to_ix = load_model_config(model_type)
    config = GRUConfig() if model_type == 'GRU' else LSTMConfig()
    model_class = GRUModel if model_type == 'GRU' else LSTMModel
    
    model = model_class(
        model_params['vocab_size'],
        model_params['tagset_size'],
        model_params['embedding_dim'],
        model_params['hidden_dim'],
        config.num_layers,
        config.dropout
    )
    model.load_state_dict(torch.load(f'Models/crf_pos_{model_type}.pth', map_location=torch.device('cpu')))
    model.eval()
    logger.info(f"{model_type} model loaded successfully")
    return model, word_to_ix, tag_to_ix, config

def load_test_data():
    logger.info("Loading test data")
    with open('test_data.json', 'r') as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test sentences")
    return test_data

def evaluate_model(model, test_data, word_to_ix, tag_to_ix, config):
    
    device = torch.device("cuda" if config.cuda else "cpu")
    model = model.to(device)
    
    logger.info("Starting model evaluation")
    correct_tags = 0
    total_tags = 0
    
    for entry in test_data:
        sentence = entry['sentence']
        real_tags = entry['tags']
        
        # Predict tags using the CRF model
        predicted_tags = infer_sentence(model, sentence, word_to_ix, tag_to_ix, config.cuda)
        predicted_tags = [tag.split('<')[1][:-1] for tag in predicted_tags.split()]
        
        # Compare predicted tags with real tags
        for real_tag, predicted_tag in zip(real_tags, predicted_tags):
            if real_tag == predicted_tag:
                correct_tags += 1
            total_tags += 1

    # Calculate accuracy
    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
    return accuracy

# Load evaluation metrics
def load_evaluation_metrics(model_type):
    logger.info(f"Loading evaluation metrics for {model_type} model")
    with open(f'Models_Metrics/{model_type}_overall_performance_metrics.json', 'r') as f:
        overall_metrics = json.load(f)
    with open(f'Models_Metrics/{model_type}_per_pos_performance_metrics.json', 'r') as f:
        per_pos_metrics = json.load(f)
    with open(f'Models_Metrics/{model_type}_most_mismatched_tags.json', 'r') as f:
        mismatched_tags = json.load(f)
    confusion_matrix = np.load(f'Models_Metrics/{model_type}_confusion_matrix.npy')
    logger.info(f"Evaluation metrics for {model_type} model loaded successfully")
    return overall_metrics, per_pos_metrics, mismatched_tags, confusion_matrix

# Streamlit app
def main():
    st.set_page_config(page_title="CRF POS App")
    st.title("CRF-based POS Tagging System")
    logger.info("Streamlit app started")
    
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_type = st.sidebar.radio("Choose a model:", ("GRU", "LSTM"))
    logger.info(f"User selected {model_type} model")
    
    # Load the selected model
    model, word_to_ix, tag_to_ix, config = load_model(model_type)
    
    # Main content area
    st.header(f"Using {model_type} Model")
    
    # Text input for sentence tagging
    st.subheader("Tag a Sentence")
    sentence = st.text_input("Enter a sentence to tag:")
    if st.button("Tag Sentence"):
        if sentence:
            logger.info(f"Tagging sentence: {sentence}")
            tagged_sentence = infer_sentence(model, sentence, word_to_ix, tag_to_ix, config.cuda)
            st.markdown("**Tagged Sentence:**")
            st.write(tagged_sentence)
            logger.info(f"Tagged sentence: {tagged_sentence}")
        else:
            st.warning("Please enter a sentence to tag.")
            logger.warning("User attempted to tag an empty sentence")

    # Model Evaluation
    st.header("Model Evaluation")
    if st.button("Start Evaluation"):
        logger.info("User initiated model evaluation")
        test_data = load_test_data()
        
        # Evaluate both models
        gru_model, gru_word_to_ix, gru_tag_to_ix, gru_config = load_model("GRU")
        lstm_model, lstm_word_to_ix, lstm_tag_to_ix, lstm_config = load_model("LSTM")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        gru_model = gru_model.to(device)
        lstm_model = lstm_model.to(device)
        
        gru_accuracy = evaluate_model(gru_model, test_data, gru_word_to_ix, gru_tag_to_ix, gru_config)
        lstm_accuracy = evaluate_model(lstm_model, test_data, lstm_word_to_ix, lstm_tag_to_ix, lstm_config)
        
        st.subheader("Evaluation Results")
        st.write(f"GRU Model Accuracy: {gru_accuracy:.4f}")
        st.write(f"LSTM Model Accuracy: {lstm_accuracy:.4f}")
        
        # Visualization
        fig, ax = plt.subplots()
        models = ['GRU', 'LSTM']
        accuracies = [gru_accuracy, lstm_accuracy]
        ax.bar(models, accuracies)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Comparison')
        st.pyplot(fig)
    
    # Display evaluation metrics
    st.header("Evaluation Metrics")
    overall_metrics, per_pos_metrics, mismatched_tags, confusion_matrix = load_evaluation_metrics(model_type)
    
    # Overall metrics
    st.subheader("Overall Performance")
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'F0.5 Score', 'F2 Score'],
        'Value': [
            overall_metrics['precision'],
            overall_metrics['recall'],
            overall_metrics['f1'],
            overall_metrics['f0.5'],
            overall_metrics['f2']
        ]
    })
    st.table(metrics_df.set_index('Metric'))
    logger.info("Displayed overall performance metrics")
    
    # Per-POS metrics
    st.subheader("Performance by POS Tag")
    pos_metrics_df = pd.DataFrame(per_pos_metrics).transpose()
    pos_metrics_df = pos_metrics_df.sort_values('f1', ascending=False)
    st.dataframe(pos_metrics_df)
    logger.info("Displayed per-POS performance metrics")
    
    # Most mismatched tags
    st.subheader("Top 5 Most Mismatched Tags")
    mismatched_df = pd.DataFrame(mismatched_tags, columns=['Count', 'True Tag', 'Predicted Tag'])
    st.table(mismatched_df)
    logger.info("Displayed top 5 most mismatched tags")
    
    # Confusion matrix heatmap
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix, ax=ax, cmap='YlOrRd', fmt='d')
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)
    logger.info("Displayed confusion matrix heatmap")

if __name__ == "__main__":
    main()