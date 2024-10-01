import os
import json
import torch
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from crf_training_GRU import CRFModel, Config, prepare_data
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, fbeta_score

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want, logging to both a file and the console."""
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

logger = setup_logger('crf_eval_gru', 'Logs/crf_pos_GRU_Evaluation_Log.log')

# Check for GPU availability
use_cuda = torch.cuda.is_available()
logger.info(f"Using CUDA: {use_cuda}")

config = Config()

# Load the trained model
model_path = 'Models/best_crf_pos_GRU.pth'

# Evaluate the model
def evaluate_model(tagged_sentences, model, word_to_ix, tag_to_ix, use_cuda):
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    all_tags = [ix_to_tag[i] for i in range(len(tag_to_ix))]

    y_true = []
    y_pred = []

    total_sentences = len(tagged_sentences)
    logger.info(f"Starting evaluation for {total_sentences} sentences...")

    for idx, (sentence_indices, true_tag_indices) in enumerate(tagged_sentences):
        sentence_tensor = pad_sequence([torch.tensor(sentence_indices)], batch_first=True)

        with torch.no_grad():
            if use_cuda:
                sentence_tensor = sentence_tensor.cuda()
            predicted_tags = model(sentence_tensor)
            predicted_tags = predicted_tags[0]

        true_tags = [ix_to_tag[idx] for idx in true_tag_indices]
        predicted_tags = [ix_to_tag[idx] for idx in predicted_tags]

        y_true.extend(true_tags)
        y_pred.extend(predicted_tags)

        # Log progress every 100 sentences or after processing all sentences
        if (idx + 1) % 100 == 0 or (idx + 1) == total_sentences:
            progress_percentage = (idx + 1) / total_sentences * 100
            logger.info(f"Progress: {progress_percentage:.2f}% ({idx + 1} out of {total_sentences} sentences processed)")

    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, average='weighted', zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted', zero_division=0)

    overall_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f0.5': f05,
        'f2': f2
    }

    logger.info(f"Overall Metrics: {overall_metrics}")

    # Calculate per-tag metrics
    per_tag_metrics = {}
    for tag in all_tags:
        tag_precision, tag_recall, tag_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[tag], average=None, zero_division=0
        )
        per_tag_metrics[tag] = {
            'precision': tag_precision[0],  # Average None returns arrays; extract first element
            'recall': tag_recall[0],
            'f1': tag_f1[0]
        }
        logger.info(f"Tag: {tag} | Precision: {tag_precision[0]:.2f} | Recall: {tag_recall[0]:.2f} | F1: {tag_f1[0]:.2f}")

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_tags)
    logger.info("Confusion matrix computed.")

    # Find most mismatched tags
    mismatches = []
    for i, tag1 in enumerate(all_tags):
        for j, tag2 in enumerate(all_tags):
            if i != j:
                mismatches.append((cm[i, j], tag1, tag2))

    mismatches.sort(reverse=True)
    most_mismatched_tags = mismatches[:5]

    logger.info("Most mismatched tags identified.")

    # Return all_tags along with the results
    return overall_metrics, per_tag_metrics, cm, most_mismatched_tags, all_tags

# Save metrics and visualizations
def save_metrics(overall_metrics, per_tag_metrics, cm, most_mismatched_tags, all_tags, model_type):
    logger.info(f"Saving metrics for model type: {model_type}")
    
    os.makedirs('Models_Metrics', exist_ok=True)
    
    # Save overall metrics
    with open(f'Models_Metrics/{model_type}_overall_performance_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    logger.info(f"Overall metrics saved to 'Models_Metrics/{model_type}_overall_performance_metrics.json'.")
    
    # Save per-tag metrics
    with open(f'Models_Metrics/{model_type}_per_pos_performance_metrics.json', 'w') as f:
        json.dump(per_tag_metrics, f, indent=4)
    logger.info(f"Per-tag metrics saved to 'Models_Metrics/{model_type}_per_pos_performance_metrics.json'.")
    
    # Save confusion matrix
    np.save(f'Models_Metrics/{model_type}_confusion_matrix.npy', cm)
    logger.info(f"Confusion matrix saved to 'Models_Metrics/{model_type}_confusion_matrix.npy'.")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_tags, yticklabels=all_tags)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'Models_Metrics/{model_type}_confusion_matrix.png')
    plt.close()
    logger.info(f"Confusion matrix plot saved as 'Models_Metrics/{model_type}_confusion_matrix.png'.")
    
    # Save most mismatched tags
    most_mismatched_tags_serializable = [
        (int(tag_count), tag1, tag2) for tag_count, tag1, tag2 in most_mismatched_tags
    ]
    with open(f'Models_Metrics/{model_type}_most_mismatched_tags.json', 'w') as f:
        json.dump(most_mismatched_tags_serializable, f, indent=4)
    logger.info(f"Most mismatched tags saved to 'Models_Metrics/{model_type}_most_mismatched_tags.json'.")


# Main execution
if __name__ == '__main__':
    training_data, word_to_ix, tag_to_ix = prepare_data()
    model = CRFModel(len(word_to_ix), len(tag_to_ix), config.embedding_dim, config.hidden_dim)
    model.load_state_dict(torch.load(model_path))
    if config.cuda:
        model = model.cuda()
    # Assuming you have already obtained these variables from the evaluate_model function
    overall_metrics, per_tag_metrics, cm, most_mismatched_tags, all_tags = evaluate_model(training_data, model, word_to_ix, tag_to_ix, use_cuda)

    # Save metrics with the correct parameters
    save_metrics(overall_metrics, per_tag_metrics, cm, most_mismatched_tags, all_tags, "LSTM")