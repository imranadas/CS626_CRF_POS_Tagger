import streamlit as st
import torch
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Import the necessary components from your existing scripts
from crf_training_GRU import CRFModel as GRUModel, Config as GRUConfig, prepare_data as prepare_data_GRU
from crf_training_LSTM import CRFModel as LSTMModel, Config as LSTMConfig, prepare_data as prepare_data_LSTM

# Function to load model and necessary data
def load_model(model_type):
    if model_type == "GRU":
        config = GRUConfig()
        model_class = GRUModel
        prepare_data = prepare_data_GRU
        model_path = 'Models/best_crf_pos_GRU.pth'
    else:
        config = LSTMConfig()
        model_class = LSTMModel
        prepare_data = prepare_data_LSTM
        model_path = 'Models/best_crf_pos_LSTM.pth'
    
    training_data, word_to_ix, tag_to_ix = prepare_data()
    model = model_class(len(word_to_ix), len(tag_to_ix), config.embedding_dim, config.hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, word_to_ix, tag_to_ix, config

# Function to perform POS tagging
def pos_tag_sentence(model, sentence, word_to_ix, tag_to_ix, config):
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    sentence_indices = [word_to_ix.get(word.lower(), 0) for word in sentence.split()]
    sentence_tensor = torch.tensor([sentence_indices], dtype=torch.long)
    
    with torch.no_grad():
        predicted_tags = model(sentence_tensor)
    
    predicted_tags = [ix_to_tag[idx] for idx in predicted_tags[0]]
    return list(zip(sentence.split(), predicted_tags))

# Streamlit app
def main():
    st.set_page_config(page_title="CRS POS Tagger(PyTorch)", layout="wide")
    st.title("POS Tagging with CRF Models")

    # Model selection in the top bar
    model_type = st.selectbox("Select Model", ["GRU", "LSTM"])

    # Load the selected model
    model, word_to_ix, tag_to_ix, config = load_model(model_type)

    # Sidebar for input and actions
    st.sidebar.header("Actions")
    input_sentence = st.sidebar.text_input("Enter a sentence for POS tagging:")
    if st.sidebar.button("Tag Sentence"):
        if input_sentence:
            tagged_sentence = pos_tag_sentence(model, input_sentence, word_to_ix, tag_to_ix, config)
            st.subheader("Tagged Sentence")
            st.table(pd.DataFrame(tagged_sentence, columns=["Word", "POS Tag"]))
        else:
            st.warning("Please enter a sentence.")

    # Display evaluation metrics
    if st.sidebar.button("Show Evaluation Metrics"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Overall Performance Metrics")
            with open(f'Model_Metrics/{model_type}_overall_performance_metrics.json', 'r') as f:
                overall_metrics = json.load(f)
            st.table(pd.DataFrame([overall_metrics]))

            st.subheader("Per-POS Performance Metrics")
            with open(f'Model_Metrics/{model_type}_per_pos_performance_metrics.json', 'r') as f:
                per_pos_metrics = json.load(f)
            st.table(pd.DataFrame(per_pos_metrics).T)

            st.subheader("Most Mismatched Tags")
            with open(f'Model_Metrics/{model_type}_most_mismatched_tags.json', 'r') as f:
                mismatched_tags = json.load(f)
            st.table(pd.DataFrame(mismatched_tags, columns=["Count", "True Tag", "Predicted Tag"]))

        with col2:
            st.subheader("Confusion Matrix")
            confusion_matrix = np.load(f'Model_Metrics/{model_type}_confusion_matrix.npy')
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(confusion_matrix, cmap='Blues')
            ax.set_title("Confusion Matrix")
            fig.colorbar(im)
            st.pyplot(fig)

            st.subheader("Confusion Matrix (from PNG)")
            confusion_matrix_img = Image.open(f'Model_Metrics/{model_type}_confusion_matrix.png')
            st.image(confusion_matrix_img, use_column_width=True)

if __name__ == "__main__":
    main()