import os
import json
import nltk
import torch
import random
import logging
import torch.nn as nn
from TorchCRF import CRF
import torch.optim as optim
from nltk.corpus import brown
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# Download NLTK datasets if not already downloaded
nltk.download('brown')
nltk.download('universal_tagset')

# Configuration section
class Config:
    def __init__(self):
        self.epochs = 25
        self.learning_rate = 0.001
        self.batch_size = 32
        self.embedding_dim = 320
        self.hidden_dim = 640
        self.num_layers = 2
        self.dropout = 0.3
        self.cuda = torch.cuda.is_available()
        self.model_save_path = 'Models/crf_pos_GRU.pth'
        self.model_name = 'CRF_GRU'
        self.patience = 5

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

logger = setup_logger('crf_train_gru', 'Logs/crf_pos_GRU_Training_Log.log')

# Initialize stemmer
stemmer = PorterStemmer()

# Feature extraction function
def extract_features(sentence, tags):
    features = []
    for word, tag in zip(sentence, tags):
        if word in ",.!?;:\"'()[]{}«»""''…—–-":
            # Treat punctuation as a separate word
            features.append((word, word, word, word, True, tag))
        else:
            stem = stemmer.stem(word.lower())
            suffix = word[-3:] if len(word) > 3 else word
            prefix = word[:3] if len(word) > 3 else word
            features.append((word.lower(), stem, suffix, prefix, False, tag))
    logger.debug(f'Extracted features: {features}')
    return features

# Data preparation
def prepare_data():
    tagged_sentences = brown.tagged_sents(tagset='universal')
    word_to_ix = defaultdict(lambda: len(word_to_ix))
    tag_to_ix = defaultdict(lambda: len(tag_to_ix))
    word_to_ix['<PAD>'] = 0
    word_to_ix['<UNK>'] = 1
    tag_to_ix['<PAD>'] = 0

    training_data = []
    logger.info(f'Total tagged sentences in Brown corpus: {len(tagged_sentences)}')

    for idx, tagged_sentence in enumerate(tagged_sentences):
        words, tags = [], []
        for word, tag in tagged_sentence:
            if word in ",.!?;:\"'()[]{}«»""''…—–-":
                # Add punctuation as a separate word
                words.append(word)
                tags.append('.')  # Use '.' tag for punctuation in universal tagset
            else:
                words.append(word)
                tags.append(tag)
        
        features = extract_features(words, tags)
        sentence_indices = [word_to_ix[word.lower()] for word, _, _, _, _, _ in features]
        tag_indices = [tag_to_ix[tag] for _, _, _, _, _, tag in features]
        training_data.append((sentence_indices, tag_indices))

        if idx % 1000 == 0:
            logger.info(f'Processed {idx}/{len(tagged_sentences)} sentences.')

    logger.info(f'Completed data preparation. Total training samples: {len(training_data)}')
    return training_data, word_to_ix, tag_to_ix

def save_model_params(word_to_ix, tag_to_ix, config):
    # Ensure the directory exists
    os.makedirs('Model_Config', exist_ok=True)
    
    model_params = {
        'vocab_size': len(word_to_ix),
        'tagset_size': len(tag_to_ix),
        'embedding_dim': config.embedding_dim,
        'hidden_dim': config.hidden_dim
    }
    
    with open('Model_Config/GRU_model_params.json', 'w') as f:
        json.dump(model_params, f)
    
    # Save word_to_ix mapping
    with open('Model_Config/GRU_word_to_ix.json', 'w') as f:
        json.dump(word_to_ix, f)
    
    # Save tag_to_ix mapping
    with open('Model_Config/GRU_tag_to_ix.json', 'w') as f:
        json.dump(tag_to_ix, f)
    
    logger.info("Model parameters and mappings saved to Model_Config/GRU_ JSON files")

# Define the model
class CRFModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(CRFModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        gru_out, _ = self.gru(embeds)
        gru_out = self.dropout(gru_out)
        emissions = self.linear(gru_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)
        
class TrainingStats:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def update(self, train_loss, train_accuracy, val_accuracy):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

    def plot(self, model_name):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title(f'{model_name} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'Models_Metrics/{model_name}_graph.png')
        plt.close()

# Padding and masking function
def prepare_batch(batch_data):
    sentences, tags = zip(*batch_data)
    sentences_tensor = pad_sequence([torch.tensor(sentence) for sentence in sentences], batch_first=True, padding_value=0)
    tags_tensor = pad_sequence([torch.tensor(tag) for tag in tags], batch_first=True, padding_value=0)
    mask = (sentences_tensor != 0).type(torch.uint8)
    return sentences_tensor, tags_tensor, mask

# Training the model
def train_model(model, train_data, val_data, config):
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    logger.info('Starting model training...')
    best_accuracy = 0
    patience_counter = 0
    stats = TrainingStats()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        logger.info(f'Epoch {epoch + 1}/{config.epochs} started.')

        for batch_idx in range(0, len(train_data), config.batch_size):
            batch_data = train_data[batch_idx:batch_idx + config.batch_size]
            sentences_tensor, tags_tensor, mask = prepare_batch(batch_data)

            if config.cuda:
                sentences_tensor = sentences_tensor.cuda()
                tags_tensor = tags_tensor.cuda()
                mask = mask.cuda()

            model.zero_grad()
            loss = model(sentences_tensor, tags_tensor, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate training accuracy
            with torch.no_grad():
                predicted_tags = model(sentences_tensor, mask=mask)
                for pred, true, m in zip(predicted_tags, tags_tensor, mask):
                    valid_length = m.sum().item()
                    pred_tensor = torch.tensor(pred[:valid_length], device=true.device)
                    epoch_correct += (pred_tensor == true[:valid_length]).sum().item()
                    epoch_total += valid_length

            if batch_idx % (config.batch_size * 10) == 0:
                logger.info(f'Epoch {epoch + 1}/{config.epochs}, Batch {batch_idx}/{len(train_data)}, Loss: {loss.item():.4f}')

        avg_loss = epoch_loss / (len(train_data) // config.batch_size)
        train_accuracy = epoch_correct / epoch_total
        val_accuracy = validate_model(model, val_data, config)
        
        stats.update(avg_loss, train_accuracy, val_accuracy)
        
        logger.info(f'Epoch {epoch + 1}/{config.epochs} completed. '
                    f'Average Loss: {avg_loss:.4f}, '
                    f'Train Accuracy: {train_accuracy:.4f}, '
                    f'Validation Accuracy: {val_accuracy:.4f}')

        scheduler.step(val_accuracy)
        
        os.makedirs('Models', exist_ok=True)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            logger.info(f'New best model saved with accuracy: {best_accuracy * 100:.2f}%')
        else:
            patience_counter += 1
            logger.info(f'No improvement. Patience counter: {patience_counter}/{config.patience}')

        if patience_counter >= config.patience:
            logger.info('Early stopping triggered.')
            break

    logger.info(f'Training completed. Best Validation Accuracy: {best_accuracy * 100:.2f}%')
    
    os.makedirs('Models_Metrics', exist_ok=True)
    
    # Plot and save the graphs
    stats.plot(config.model_name)


# Validate the model
def validate_model(model, val_data, config):
    model.eval()
    total_correct = 0
    total_tags = 0

    with torch.no_grad():
        for batch_idx in range(0, len(val_data), config.batch_size):
            batch_data = val_data[batch_idx:batch_idx + config.batch_size]
            sentences_tensor, tags_tensor, mask = prepare_batch(batch_data)

            if config.cuda:
                sentences_tensor = sentences_tensor.cuda()
                tags_tensor = tags_tensor.cuda()
                mask = mask.cuda()

            predicted_tags = model(sentences_tensor, mask=mask)

            for pred, true, m in zip(predicted_tags, tags_tensor, mask):
                valid_length = m.sum().item()
                
                # Ensure `pred` is on the same device as `true`
                pred_tensor = torch.tensor(pred[:valid_length], device=true.device)

                total_correct += (pred_tensor == true[:valid_length]).sum().item()
                total_tags += valid_length

            logger.info(f'Validation batch {batch_idx // config.batch_size + 1}/{len(val_data) // config.batch_size + 1} processed.')

    accuracy = total_correct / total_tags if total_tags > 0 else 0
    logger.info(f'Validation Accuracy: {accuracy * 100:.2f}%')
    return accuracy

# Main function
if __name__ == "__main__":
    config = Config()
    training_data, word_to_ix, tag_to_ix = prepare_data()
    save_model_params(word_to_ix, tag_to_ix, config)

    logger.info(f'Vocabulary size: {len(word_to_ix)}')
    logger.info(f'Number of unique tags: {len(tag_to_ix)}')
    logger.info(f'Tags: {", ".join(tag_to_ix.keys())}')

    random.seed(42)
    torch.manual_seed(42)
    if config.cuda:
        torch.cuda.manual_seed(42)

    train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)

    model = CRFModel(len(word_to_ix), len(tag_to_ix), config.embedding_dim, config.hidden_dim, config.num_layers, config.dropout)
    
    logger.info(f'Training on {len(train_data)} samples and validating on {len(val_data)} samples.')
    logger.info(f'Total Word IDX: {len(word_to_ix)}, Total Tags IDX: {len(tag_to_ix)}')
    
    if config.cuda:
        model.cuda()
    
    train_model(model, train_data, val_data, config)