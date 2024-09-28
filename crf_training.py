import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import KFold
import json
import pickle

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),  # Output to console
    logging.FileHandler('crf_model_training.log')  # Save to file
])

# Download necessary NLTK resources
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')

class CRFPOSTagger:
    def __init__(self):
        self.feature_function = None
        self.weights = defaultdict(float)
        self.tags = set()

    def extract_features(self, sentence, i, prev_tag, current_tag):
        """Extract features for the current word and previous word."""
        word = sentence[i]
        features = {
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],  # Last 3 letters
            'word[-2:]': word[-2:],  # Last 2 letters
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'prev_tag': prev_tag
        }
        if i == 0:
            features["BOS"] = True  # Beginning of sentence
        elif i == len(sentence) - 1:
            features["EOS"] = True  # End of sentence
        return features

    def train(self, tagged_sentences, num_iterations=5):
        logging.info("Training CRF POS tagger...")
        if not tagged_sentences:
            logging.warning("Training data is empty. Aborting training.")
            return

        self.tags = set(tag for sent in tagged_sentences for _, tag in sent)
        for _ in range(num_iterations):
            for sent in tagged_sentences:
                words, true_tags = zip(*sent)
                predicted_tags = self.viterbi(words)
                for i in range(len(words)):
                    true_tag = true_tags[i]
                    pred_tag = predicted_tags[i]
                    if true_tag != pred_tag:
                        self.update_weights(words, i, true_tag, pred_tag)

        logging.info("Training completed.")

    def update_weights(self, sentence, i, true_tag, pred_tag):
        true_features = self.extract_features(sentence, i, '', true_tag)
        pred_features = self.extract_features(sentence, i, '', pred_tag)
        for feature in true_features:
            self.weights[feature] += 1
        for feature in pred_features:
            self.weights[feature] -= 1

    def viterbi(self, sentence):
        V = [{}]
        path = {}

        tags = list(self.tags)
        # Initialize base case
        for tag in tags:
            V[0][tag] = sum(self.weights.get(f, 0.0) for f in self.extract_features(sentence, 0, '', tag))
            path[tag] = [tag]

        # Run Viterbi algorithm for the rest of the sentence
        for i in range(1, len(sentence)):
            V.append({})
            new_path = {}
            for tag in tags:
                (prob, prev_tag) = max(
                    (V[i - 1][ptag] + sum(self.weights.get(f, 0.0) for f in self.extract_features(sentence, i, ptag, tag)), ptag)
                    for ptag in tags
                )
                V[i][tag] = prob
                new_path[tag] = path[prev_tag] + [tag]
            path = new_path

        (prob, state) = max((V[len(sentence) - 1][tag], tag) for tag in tags)
        return path[state]

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)
        logging.info(f"Model saved to {filename}.")

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
            tagger = CRFPOSTagger()
            tagger.weights = defaultdict(float, weights)
            return tagger

def preprocess_data():
    logging.info("Preprocessing data...")
    brown_sents = brown.tagged_sents(tagset='universal')
    data = []
    for idx, sent in enumerate(brown_sents):
        words, tags = zip(*sent)
        tokens = [word_tokenize(word.lower()) for word in words]
        processed_sent = list(zip([item for sublist in tokens for item in sublist], tags))
        data.append(processed_sent)
        if idx % 1000 == 0:
            logging.info(f"Preprocessed {idx + 1} sentences.")
    logging.info(f"Preprocessed {len(data)} sentences.")
    return data

def evaluate_model(tagged_sentences, tagger, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_tags = list(set(tag for _, tag in sum(tagged_sentences, [])))

    overall_precision = []
    overall_recall = []
    overall_f1 = []
    overall_f05 = []
    overall_f2 = []

    per_tag_results = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
    conf_matrix = np.zeros((len(all_tags), len(all_tags)))

    fold_number = 1
    for train_index, test_index in kf.split(tagged_sentences):
        logging.info(f"Starting fold {fold_number}...")
        train_data = [tagged_sentences[i] for i in train_index]
        test_data = [tagged_sentences[i] for i in test_index]
        
        tagger.train(train_data)

        y_true = []
        y_pred = []

        for idx, sent in enumerate(test_data):
            words, true_tags = zip(*sent)
            predicted_tags = tagger.viterbi(words)
            y_true.extend(true_tags)
            y_pred.extend(predicted_tags)
            if idx % 100 == 0:
                logging.info(f"Processed {idx + 1} sentences in test data.")

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        f05 = 1.25 * precision * recall / (0.25 * precision + recall)
        f2 = 5 * precision * recall / (4 * precision + recall)

        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        overall_f05.append(f05)
        overall_f2.append(f2)

        for tag in all_tags:
            true_tag_indices = [i for i, t in enumerate(y_true) if t == tag]
            pred_tag_indices = [i for i, t in enumerate(y_pred) if t == tag]
            p, r, f, _ = precision_recall_fscore_support([y_true[i] for i in true_tag_indices], [y_pred[i] for i in true_tag_indices], average='macro', zero_division=0)
            per_tag_results[tag]['precision'].append(p)
            per_tag_results[tag]['recall'].append(r)
            per_tag_results[tag]['f1'].append(f)

        cm = confusion_matrix(y_true, y_pred, labels=all_tags)
        conf_matrix += cm

        logging.info(f"Completed fold {fold_number}.")
        fold_number += 1

    avg_precision = np.mean(overall_precision)
    avg_recall = np.mean(overall_recall)
    avg_f1 = np.mean(overall_f1)
    avg_f05 = np.mean(overall_f05)
    avg_f2 = np.mean(overall_f2)

    overall_metrics = {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1, 'f0.5': avg_f05, 'f2': avg_f2}
    with open('overall_performance_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    logging.info("Overall performance metrics saved to 'overall_performance_metrics.json'.")

    for tag in per_tag_results:
        per_tag_results[tag] = {metric: np.mean(values) for metric, values in per_tag_results[tag].items()}

    with open('per_pos_performance.json', 'w') as f:
        json.dump(per_tag_results, f, indent=4)
    logging.info("Per-POS performance metrics saved to 'per_pos_performance.json'.")

    np.save('confusion_matrix.npy', conf_matrix)
    logging.info("Confusion matrix saved to 'confusion_matrix.npy'.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=all_tags, yticklabels=all_tags)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    logging.info("Confusion matrix plot saved as 'confusion_matrix.png'.")

    mismatches = []
    for i, tag1 in enumerate(all_tags):
        for j, tag2 in enumerate(all_tags):
            if i != j:
                mismatches.append((conf_matrix[i, j], tag1, tag2))
    
    mismatches.sort(reverse=True)
    most_mismatched_tags = mismatches[:5]
    
    with open('most_mismatched_tags.json', 'w') as f:
        json.dump(most_mismatched_tags, f, indent=4)
    logging.info("Most mismatched tags saved to 'most_mismatched_tags.json'.")

if __name__ == "__main__":
    logging.info("Starting CRF POS Tagger...")
    data = preprocess_data()
    tagger = CRFPOSTagger()
    evaluate_model(data, tagger)
    tagger.save_model('crf_pos_tagger.pkl')
