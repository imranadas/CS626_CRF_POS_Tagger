import os
import json
import torch
import logging
from crf_training_GRU import CRFModel as GRUModel, Config as GRUConfig
from crf_training_LSTM import CRFModel as LSTMModel, Config as LSTMConfig, extract_features

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

logger = setup_logger('crf_inference', 'Logs/crf_pos_inference_log.log')

def load_model_params(model_type):
    with open(f'Model_Config/{model_type}_model_params.json', 'r') as f:
        model_params = json.load(f)
    
    logger.info(f"{model_type} model parameters loaded from Model_Config/{model_type}_model_params.json")
    return model_params

def load_mappings(model_type):
    with open(f'Model_Config/{model_type}_word_to_ix.json', 'r') as f:
        word_to_ix = json.load(f)
    with open(f'Model_Config/{model_type}_tag_to_ix.json', 'r') as f:
        tag_to_ix = json.load(f)
    
    logger.info(f"{model_type} word and tag mappings loaded from JSON files")
    return word_to_ix, tag_to_ix

# Function to initialize the model
def initialize_model(model_type):
    if model_type.lower() == "gru":
        logger.info("Initializing GRU model...")
        config = GRUConfig()
        model_class = GRUModel
        model_path = 'Models/crf_pos_GRU.pth'
    elif model_type.lower() == "lstm":
        logger.info("Initializing LSTM model...")
        config = LSTMConfig()
        model_class = LSTMModel
        model_path = 'Models/crf_pos_LSTM.pth'
    else:
        raise ValueError("Invalid model type. Choose between 'GRU' or 'LSTM'.")
    
    # Load model parameters
    model_params = load_model_params(model_type.upper())
    
    # Load word and tag mappings
    word_to_ix, tag_to_ix = load_mappings(model_type.upper())
    
    # Initialize model
    model = model_class(
        model_params['vocab_size'],
        model_params['tagset_size'],
        model_params['embedding_dim'],
        model_params['hidden_dim'],
        config.num_layers,
        config.dropout
    )
    model.load_state_dict(torch.load(model_path))
    
    # If CUDA is available, move model to GPU
    if config.cuda:
        model = model.cuda()

    logger.info(f"{model_type.upper()} model loaded successfully from {model_path}")
    return model, word_to_ix, tag_to_ix, config

# Function to process input sentence and predict tags
def infer_sentence(model, sentence, word_to_ix, tag_to_ix, use_cuda):
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    
    # Split the sentence into words and keep punctuation separate
    words = []
    for word in sentence.split():
        if word[-1] in ",.!?;:\"'()[]{}«»""''…—–-":
            if len(word[:-1]) > 0:  # Only add if the word part exists
                words.append(word[:-1])
            words.append(word[-1])
        else:
            words.append(word)

    # Generate dummy tags for the features function
    dummy_tags = ['DUMMY'] * len(words)
    features = extract_features(words, dummy_tags)
    
    # Convert words to indices, use <UNK> for unknown words
    sentence_indices = [word_to_ix.get(word.lower(), word_to_ix['<UNK>']) for word, _, _, _, _, _ in features]
    sentence_tensor = torch.tensor([sentence_indices], dtype=torch.long).to(device)
    
    if use_cuda:
        sentence_tensor = sentence_tensor.cuda()
    
    with torch.no_grad():
        predicted_tags = model(sentence_tensor)
    
    predicted_tags = [ix_to_tag[idx] for idx in predicted_tags[0]]
    
    # Return a list of tuples containing words and their predicted tags
    return list(zip(words, predicted_tags))

# Main inferencing loop
def run_inferencing_loop():
    model_type = input("Enter the model to initialize (GRU/LSTM): ").strip().lower()
    
    try:
        model, word_to_ix, tag_to_ix, config = initialize_model(model_type)
    except ValueError as e:
        logger.error(e)
        return

    logger.info("Starting inference loop. Type 'exit' to quit.")

    while True:
        sentence = input("Enter a sentence: ").strip()
        
        if sentence.lower() == 'exit':
            logger.info("Exiting inference loop.")
            break
        
        logger.info(f"Input Sentence: {sentence}")
        
        result = infer_sentence(model, sentence, word_to_ix, tag_to_ix, config.cuda)
        formatted_result = " ".join([f"{word}<{tag}>" for word, tag in result])
        logger.info(f"Output Tagged Sentence: {formatted_result}")
        print(formatted_result)

# Entry point of the script
if __name__ == "__main__":
    run_inferencing_loop()
