import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from crf_inference import initialize_model, infer_sentence
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Set page configuration
st.set_page_config(page_title="CRF POS Tagger", layout="wide")

# Set up the sidebar
st.sidebar.title("CRF POS Tagger")
model_type = st.sidebar.selectbox("Select Model", ["LSTM", "GRU"])

# Load the model and mappings
model, word_to_ix, tag_to_ix, config = initialize_model(model_type)

@st.cache_data
def load_test_data():
    with open("test_data.json", "r") as f:
        test_data = json.load(f)
    return test_data

# Main page layout
st.title("CRF POS Tagger")

# Sentence input and prediction
st.header("Input Sentence")
input_sentence = st.text_area("Enter a sentence", height=100)

if st.button("Predict Tags"):
    result = infer_sentence(model, input_sentence, word_to_ix, tag_to_ix, config.cuda)
    st.write("Tagged Sentence:")
    st.code(result)

# Evaluation metrics tabs
st.header("Evaluation Metrics")
tabs = st.tabs(["Overall Metrics", "Per-Tag Metrics", "Confusion Matrix", "Mismatched Tags"])

with tabs[0]:
    st.subheader("Overall Metrics")
    overall_metrics_path = f"Models_Metrics/{model_type.lower()}_overall_performance_metrics.json"
    if os.path.exists(overall_metrics_path):
        overall_metrics = pd.DataFrame.from_dict(json.load(open(overall_metrics_path, "r")), orient="index")
        st.dataframe(overall_metrics)
    else:
        st.write("No overall metrics available. Please run the evaluation script first.")

with tabs[1]:
    st.subheader("Per-Tag Metrics")
    per_tag_metrics_path = f"Models_Metrics/{model_type.lower()}_per_pos_performance_metrics.json"
    if os.path.exists(per_tag_metrics_path):
        per_tag_metrics = pd.DataFrame.from_dict(json.load(open(per_tag_metrics_path, "r")), orient="index")
        st.dataframe(per_tag_metrics)
    else:
        st.write("No per-tag metrics available. Please run the evaluation script first.")

with tabs[2]:
    st.subheader("Confusion Matrix")
    confusion_matrix_path = f"Models_Metrics/{model_type.lower()}_confusion_matrix.npy"
    if os.path.exists(confusion_matrix_path):
        cm = np.load(confusion_matrix_path)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(tag_to_ix.keys()), yticklabels=list(tag_to_ix.keys()))
        plt.title("Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        st.pyplot(fig)
    else:
        st.write("No confusion matrix available. Please run the evaluation script first.")

with tabs[3]:
    st.subheader("Most Mismatched Tags")
    most_mismatched_tags_path = f"Models_Metrics/{model_type.lower()}_most_mismatched_tags.json"
    if os.path.exists(most_mismatched_tags_path):
        most_mismatched_tags = pd.DataFrame(json.load(open(most_mismatched_tags_path, "r")), columns=["Count", "Tag1", "Tag2"])
        st.dataframe(most_mismatched_tags)
    else:
        st.write("No most mismatched tags available. Please run the evaluation script first.")

# Test data evaluation
st.header("Test Data Evaluation")
test_data = load_test_data()

if st.button("Evaluate on Test Data"):
    true_tags = []
    pred_tags = []
    for sample in test_data:
        sentence = sample["sentence"]
        result = infer_sentence(model, sentence, word_to_ix, tag_to_ix, config.cuda)
        pred_tags_list = [tag.split("<")[1][:-1] for tag in result.split()]
        true_tags.extend(sample["tags"])
        pred_tags.extend(pred_tags_list)

    # Ensure true_tags and pred_tags have the same length
    min_length = min(len(true_tags), len(pred_tags))
    true_tags = true_tags[:min_length]
    pred_tags = pred_tags[:min_length]

    accuracy = accuracy_score(true_tags, pred_tags)
    precision, recall, f1, _ = precision_recall_fscore_support(true_tags, pred_tags, average='weighted')

    st.write(f"Test Data Accuracy: {accuracy:.2%}")
    st.write(f"Test Data Precision: {precision:.2%}")
    st.write(f"Test Data Recall: {recall:.2%}")
    st.write(f"Test Data F1-Score: {f1:.2%}")

