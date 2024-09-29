import logging
import torch
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from nltk.stem import PorterStemmer
from crf_training_GRU import CRFModel, Config

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
nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()

# Check for GPU availability
use_cuda = torch.cuda.is_available()
logger.info(f"Using CUDA: {use_cuda}")

config = Config()

# Load the trained model
model_path = 'Models/best_crf_pos_GRU.pth'

# Data preparation function
def extract_features(sentence, tags):
    """Extracts features from the input sentence and its corresponding tags."""
    features = []
    for word, tag in zip(sentence, tags):
        stem = stemmer.stem(word.lower())
        suffix = word[-2:] if len(word) > 2 else word
        features.append((word.lower(), stem, suffix, tag))
    return features

def prepare_data():
    """Prepares training data from the Brown corpus."""
    tagged_sentences = brown.tagged_sents(tagset='universal')

    word_to_ix = defaultdict(lambda: len(word_to_ix))
    tag_to_ix = defaultdict(lambda: len(tag_to_ix))

    training_data = []
    logger.info(f'Total tagged sentences in Brown corpus: {len(tagged_sentences)}')

    for idx, tagged_sentence in enumerate(tagged_sentences):
        words = [word for word, tag in tagged_sentence]
        tags = [tag for word, tag in tagged_sentence]
        features = extract_features(words, tags)

        sentence_indices = [word_to_ix[word.lower()] for word, stem, suffix, tag in features]
        tag_indices = [tag_to_ix[tag] for word, stem, suffix, tag in features]
        training_data.append((sentence_indices, tag_indices))

        if idx % 1000 == 0:
            logger.info(f'Processed {idx}/{len(tagged_sentences)} sentences.')

    logger.info(f'Completed data preparation. Total training samples: {len(training_data)}')
    return training_data, word_to_ix, tag_to_ix

# Evaluate the model
def evaluate_model(tagged_sentences, model, word_to_ix, use_cuda):
    # Create a list of unique tags from the true tags
    all_tags = list(set(tag for sentence in tagged_sentences for tag in sentence[1]))
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

    # Evaluate on the tagged sentences
    for sent in tagged_sentences:
        sentence_indices, true_tag_indices = sent  # Unpack the word and tag indices
        sentence_tensor = pad_sequence([torch.tensor(sentence_indices)], batch_first=True)

        with torch.no_grad():
            if use_cuda:
                sentence_tensor = sentence_tensor.cuda()
            predicted_tags = model(sentence_tensor)  # Get model predictions
            predicted_tags = predicted_tags.argmax(dim=1).cpu().numpy()

        y_true.extend(true_tag_indices)
        y_pred.extend(predicted_tags)

        logger.info("Processed sentence indices: %s | True tags: %s | Predicted tags: %s", 
                    sentence_indices, true_tag_indices, predicted_tags)

    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    overall_precision.append(precision)
    overall_recall.append(recall)
    overall_f1.append(f1)
    overall_f05.append(f05)
    overall_f2.append(f2)

    logger.info("Overall Precision: %.2f | Recall: %.2f | F1: %.2f | F0.5: %.2f | F2: %.2f", 
                precision, recall, f1, f05, f2)

    # Calculate per-tag metrics
    for tag in all_tags:
        true_tag_indices = [i for i, t in enumerate(y_true) if t == tag]
        pred_tag_indices = [i for i, t in enumerate(y_pred) if t == tag]

        if true_tag_indices and pred_tag_indices:
            p, r, f, _ = precision_recall_fscore_support(
                [y_true[i] for i in true_tag_indices],
                [y_pred[i] for i in true_tag_indices],
                average='binary',  # Change to 'binary' if evaluating one tag at a time
                zero_division=0
            )
        else:
            p, r, f = 0, 0, 0

        per_tag_results[tag]['precision'].append(p)
        per_tag_results[tag]['recall'].append(r)
        per_tag_results[tag]['f1'].append(f)
        logger.info("Tag: %s | Precision: %.2f | Recall: %.2f | F1: %.2f", tag, p, r, f)

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
    
    with open('GRU_overall_performance_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    logger.info("Overall performance metrics saved to 'GRU_overall_performance_metrics.json'.")

    # Save per-POS performance metrics
    per_tag_metrics = {
        tag: {
            'precision': np.mean(results['precision']),
            'recall': np.mean(results['recall']),
            'f1': np.mean(results['f1'])
        }
        for tag, results in per_tag_results.items()
    }
    
    with open('GRU_per_pos_performance_metrics.json', 'w') as f:
        json.dump(per_tag_metrics, f, indent=4)
    logger.info("Per-POS performance metrics saved to 'GRU_per_pos_performance_metrics.json'.")

    # Save confusion matrix
    np.save('GRU_confusion_matrix.npy', conf_matrix)
    logger.info("Confusion matrix saved to 'GRU_confusion_matrix.npy'.")

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=all_tags, yticklabels=all_tags)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('GRU_confusion_matrix.png')  # Save plot to file
    plt.show()
    logger.info("Confusion matrix plot saved as 'GRU_confusion_matrix.png'.")

    # Find most mismatched tags
    mismatches = []
    for i, tag1 in enumerate(all_tags):
        for j, tag2 in enumerate(all_tags):
            if i != j:
                mismatches.append((conf_matrix[i, j], tag1, tag2))

    mismatches.sort(reverse=True)
    most_mismatched_tags = mismatches[:5]  # Top 5 mismatches

    # Save most mismatched tags
    with open('GRU_most_mismatched_tags.json', 'w') as f:
        json.dump(most_mismatched_tags, f, indent=4)
    logger.info("Most mismatched tags saved to 'GRU_most_mismatched_tags.json'.")


# Main execution
if __name__ == '__main__':
    training_data, word_to_ix, tag_to_ix = prepare_data()
    model = CRFModel(len(word_to_ix), len(tag_to_ix), config.embedding_dim, config.hidden_dim)
    model.load_state_dict(torch.load(model_path))
    if config.cuda:
        model = model.cuda()
    evaluate_model(training_data, model, word_to_ix, use_cuda)