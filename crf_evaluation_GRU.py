import nltk
import json
import torch
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from crf_training_GRU import CRFModel, Config, prepare_data
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('Logs/crf_pos_GRU_Evaluation_Log.log')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Download NLTK Brown dataset if not already downloaded
nltk.download('brown')
nltk.download('universal_tagset')

# Check for GPU availability
use_cuda = torch.cuda.is_available()
logger.info(f"Using CUDA: {use_cuda}")

config = Config()

# Load the trained model
model_path = 'Models/best_crf_pos_GRU.pth'

# Evaluate the model
def evaluate_model(tagged_sentences, model, word_to_ix, tag_to_ix, use_cuda):
    # Create reverse mapping for tag indices to tag names
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    
    # Create a list of unique tags from the true tags
    all_tags = [ix_to_tag[i] for i in range(len(tag_to_ix))]
    logger.info("Evaluating model with tags: %s", all_tags)

    # Initialize lists for overall metrics
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    overall_f05 = []
    overall_f2 = []
    
    per_tag_results = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
    conf_matrix = np.zeros((len(all_tags), len(all_tags)))

    # Prepare lists to collect true and predicted tags
    y_true = []
    y_pred = []

    total_sentences = len(tagged_sentences)

    # Evaluate on the tagged sentences
    for idx, sent in enumerate(tagged_sentences):
        sentence_indices, true_tag_indices = sent
        sentence_tensor = pad_sequence([torch.tensor(sentence_indices)], batch_first=True)

        with torch.no_grad():
            if use_cuda:
                sentence_tensor = sentence_tensor.cuda()
            predicted_tags = model(sentence_tensor)
            predicted_tags = predicted_tags[0]

        # Convert true and predicted indices to tag names
        true_tags = [ix_to_tag[idx] for idx in true_tag_indices]
        predicted_tags = [ix_to_tag[idx] for idx in predicted_tags]

        y_true.extend(true_tags)
        y_pred.extend(predicted_tags)

        logger.debug("Processed sentence %d/%d | True tags: %s | Predicted tags: %s",
                    idx + 1, total_sentences, true_tags, predicted_tags)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_sentences:
            progress_percentage = (idx + 1) / total_sentences * 100
            logger.info("Progress: %.2f%% (%d out of %d sentences processed)",
                        progress_percentage, idx + 1, total_sentences)

    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    overall_precision.append(precision)
    overall_recall.append(recall)
    overall_f1.append(f1)
    overall_f05.append(f05)
    overall_f2.append(f2)

    logger.info("Overall Precision: %.2f | Recall: %.2f | F1: %.2f | F0.5: %.2f | F2: %.2f", 
                precision, recall, f1, f05, f2)

    # Update per-tag metrics calculation
    for tag_idx in range(len(all_tags)):
        tag_name = ix_to_tag[tag_idx]
        true_tag_indices = [i for i, t in enumerate(y_true) if t == ix_to_tag[tag_idx]]
        pred_tag_indices = [i for i, t in enumerate(y_pred) if t == ix_to_tag[tag_idx]]

        if true_tag_indices and pred_tag_indices:
            p, r, f, _ = precision_recall_fscore_support(
                [y_true[i] for i in true_tag_indices],
                [y_pred[i] for i in true_tag_indices],
                average='micro',
                zero_division=0
            )
        else:
            p, r, f = 0, 0, 0

        per_tag_results[tag_name]['precision'].append(p)
        per_tag_results[tag_name]['recall'].append(r)
        per_tag_results[tag_name]['f1'].append(f)
        logger.info("Tag: %s | Precision: %.2f | Recall: %.2f | F1: %.2f", tag_name, p, r, f)

    # Create and save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_tags)
    conf_matrix += cm
    logger.info("Confusion matrix computed.")

    # Save overall performance metrics
    overall_metrics = {
        'precision': np.mean(overall_precision),
        'recall': np.mean(overall_recall),
        'f1': np.mean(overall_f1),
        'f0.5': np.mean(overall_f05),
        'f2': np.mean(overall_f2)
    }
    
    with open('Model_Metrics/GRU_overall_performance_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    logger.info("Overall performance metrics saved to 'Model_Metrics/GRU_overall_performance_metrics.json'.")
    
    # Save per-POS performance metrics
    per_tag_metrics = {
        tag: {
            'precision': np.mean(results['precision']),
            'recall': np.mean(results['recall']),
            'f1': np.mean(results['f1'])
        }
        for tag, results in per_tag_results.items()
    }
    
    with open('Model_Metrics/GRU_per_pos_performance_metrics.json', 'w') as f:
        json.dump(per_tag_metrics, f, indent=4)
    logger.info("Per-POS performance metrics saved to 'Model_Metrics/GRU_per_pos_performance_metrics.json'.")

    # Save confusion matrix
    np.save('Model_Metrics/GRU_confusion_matrix.npy', conf_matrix)
    logger.info("Confusion matrix saved to 'Model_Metrics/GRU_confusion_matrix.npy'.")

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=all_tags, yticklabels=all_tags)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Model_Metrics/GRU_confusion_matrix.png')
    plt.show()
    logger.info("Confusion matrix plot saved as 'Model_Metrics/GRU_confusion_matrix.png'.")

    # Find most mismatched tags
    mismatches = []
    for i, tag1 in enumerate(all_tags):
        for j, tag2 in enumerate(all_tags):
            if i != j:
                mismatches.append((conf_matrix[i, j], tag1, tag2))

    mismatches.sort(reverse=True)
    most_mismatched_tags = mismatches[:5]

    # Save most mismatched tags
    with open('Model_Metrics/GRU_most_mismatched_tags.json', 'w') as f:
        json.dump(most_mismatched_tags, f, indent=4)
    logger.info("Most mismatched tags saved to 'Model_Metrics/GRU_most_mismatched_tags.json'.")

# Main execution
if __name__ == '__main__':
    training_data, word_to_ix, tag_to_ix = prepare_data()
    model = CRFModel(len(word_to_ix), len(tag_to_ix), config.embedding_dim, config.hidden_dim)
    model.load_state_dict(torch.load(model_path))
    if config.cuda:
        model = model.cuda()
    evaluate_model(training_data, model, word_to_ix, tag_to_ix, use_cuda)