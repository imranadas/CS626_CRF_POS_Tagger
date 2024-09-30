import nltk
import torch
import random
import logging
import torch.nn as nn
from TorchCRF import CRF
import torch.optim as optim
from nltk.corpus import brown
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
        self.batch_size = 128
        self.embedding_dim = 256
        self.hidden_dim = 512
        self.cuda = torch.cuda.is_available()
        self.model_save_path = 'crf_pos_LSTM.pth'
        self.patience = 5

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('Logs/crf_pos_LSTM_Training_Log.log')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize stemmer
stemmer = PorterStemmer()

# Feature extraction function
def extract_features(sentence, tags):
    features = []
    for word, tag in zip(sentence, tags):
        stem = stemmer.stem(word.lower())
        suffix = word[-2:] if len(word) > 2 else word
        features.append((word.lower(), stem, suffix, tag))
    logger.debug(f'Extracted features: {features}')
    return features

# Data preparation
def prepare_data():
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

# Define the model
class CRFModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(CRFModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.linear(lstm_out)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)

# Padding and masking function
def prepare_batch(batch_data):
    sentences, tags = zip(*batch_data)
    sentences_tensor = pad_sequence([torch.tensor(sentence) for sentence in sentences], batch_first=True)
    tags_tensor = pad_sequence([torch.tensor(tag) for tag in tags], batch_first=True)

    mask = (sentences_tensor != 0).type(torch.uint8)
    mask[:, 0] = 1
    
    return sentences_tensor, tags_tensor, mask

# Training the model
def train_model(model, train_data, val_data, config):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    logger.info('Starting model training...')

    best_accuracy = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        logger.info(f'Starting epoch {epoch + 1}/{config.epochs}')
        
        epoch_loss = 0
        
        # Process data in batches
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
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % (config.batch_size * 10) == 0:
                logger.info(f'Epoch {epoch + 1}/{config.epochs}, Batch {batch_idx}/{len(train_data)}, Loss: {loss.item()}')

        avg_loss = epoch_loss / (len(train_data) // config.batch_size)
        logger.info(f'End of epoch {epoch + 1}/{config.epochs}, Average Loss: {avg_loss:.4f}')

        # Validate after each epoch
        val_accuracy = validate_model(model, val_data, config)
        
        # Check for improvement
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_' + config.model_save_path)
            logger.info(f'New best model saved with accuracy: {best_accuracy * 100:.2f}%')
        else:
            patience_counter += 1
            logger.info(f'No improvement. Patience counter: {patience_counter}/{config.patience}')

        if patience_counter >= config.patience:
            logger.info('Early stopping triggered.')
            break

    # Always save the final model after training
    torch.save(model.state_dict(), 'final_' + config.model_save_path)
    logger.info(f'Final model saved at {config.model_save_path}')

    logger.info(f'Training completed. Best Validation Accuracy: {best_accuracy * 100:.2f}%')

# Validate the model
def validate_model(model, val_data, config):
    model.eval()
    total_correct = 0
    total_tags = 0

    logger.info('Starting validation...')
    
    with torch.no_grad():
        for sentence, tags in val_data:
            # Prepare the batch for a single sample
            sentence_tensor, tags_tensor, mask = prepare_batch([(sentence, tags)])

            if config.cuda:
                sentence_tensor = sentence_tensor.cuda()
                tags_tensor = tags_tensor.cuda()
                mask = mask.cuda()

            predicted_tags = model(sentence_tensor, mask=mask)

            # Convert predicted_tags to tensor if it's a list
            if isinstance(predicted_tags, list):
                predicted_tags = torch.tensor(predicted_tags[0]).cuda() if config.cuda else torch.tensor(predicted_tags[0])

            # Remove padding from both predicted_tags and tags_tensor
            valid_length = mask.sum().item()
            predicted_tags = predicted_tags[:valid_length]
            actual_tags = tags_tensor[0][:valid_length]

            # Ensure both predicted_tags and actual_tags have compatible shapes
            total_correct += (predicted_tags == actual_tags).sum().item()
            total_tags += len(actual_tags)

    accuracy = total_correct / total_tags if total_tags > 0 else 0
    logger.info(f'Validation completed. Total Correct: {total_correct}, Total Tags: {total_tags}, Validation Accuracy: {accuracy * 100:.2f}%')
    return accuracy


# Main function
if __name__ == "__main__":
    config = Config()
    training_data, word_to_ix, tag_to_ix = prepare_data()
    
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if config.cuda:
        torch.cuda.manual_seed_all(42)

    # Split the data into training and validation sets with a random state
    train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)

    logger.info(f'Training on {len(train_data)} samples and validating on {len(val_data)} samples.')
    
    logger.info(f'Total Word IDX: {len(word_to_ix)}, Total Tags IDX: {len(tag_to_ix)}')

    model = CRFModel(len(word_to_ix), len(tag_to_ix), config.embedding_dim, config.hidden_dim)
    
    if config.cuda:
        model = model.cuda()

    train_model(model, train_data, val_data, config)
