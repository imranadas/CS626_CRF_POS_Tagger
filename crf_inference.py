import torch
import logging
from collections import defaultdict
from crf_training_GRU import CRFModel as GRUModel, Config as GRUConfig, prepare_data as prepare_data_GRU
from crf_training_LSTM import CRFModel as LSTMModel, Config as LSTMConfig, prepare_data as prepare_data_LSTM

# Function to reset logging handlers
def reset_logging():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    return logger

# Set up logging (clearing previous handlers first)
logger = reset_logging()
logger.setLevel(logging.INFO)

# Console handler for logging
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler for logging
file_handler = logging.FileHandler('Logs/crf_pos_inference_log.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Function to initialize the model
def initialize_model(model_type):
    if model_type.lower() == "gru":
        logger.info("Initializing GRU model...")
        config = GRUConfig()
        model_class = GRUModel
        prepare_data = prepare_data_GRU
        model_path = 'Models/best_crf_pos_GRU.pth'
    elif model_type.lower() == "lstm":
        logger.info("Initializing LSTM model...")
        config = LSTMConfig()
        model_class = LSTMModel
        prepare_data = prepare_data_LSTM
        model_path = 'Models/best_crf_pos_LSTM.pth'
    else:
        raise ValueError("Invalid model type. Choose between 'GRU' or 'LSTM'.")
    
    # Prepare the data (to get word_to_ix and tag_to_ix)
    training_data, word_to_ix, tag_to_ix = prepare_data()
    
    # Initialize model
    model = model_class(len(word_to_ix), len(tag_to_ix), config.embedding_dim, config.hidden_dim)
    model.load_state_dict(torch.load(model_path))
    
    # If CUDA is available, move model to GPU
    if config.cuda:
        model = model.cuda()

    logger.info(f"{model_type.upper()} model loaded successfully from {model_path}")
    return model, word_to_ix, tag_to_ix, config

# Function to process input sentence and predict tags
def infer_sentence(model, sentence, word_to_ix, tag_to_ix, use_cuda):
    # Create reverse mapping for tag indices to tag names
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    
    # Tokenize input sentence and extract features
    sentence_indices = [word_to_ix.get(word.lower(), 0) for word in sentence.split()]
    sentence_tensor = torch.tensor([sentence_indices], dtype=torch.long)

    # Move to GPU if CUDA is enabled
    if use_cuda:
        sentence_tensor = sentence_tensor.cuda()
    
    # Inference
    with torch.no_grad():
        predicted_tags = model(sentence_tensor)
    
    # Convert predicted indices to tag names
    predicted_tags = [ix_to_tag[idx] for idx in predicted_tags[0]]
    
    # Return words with predicted tags
    result = " ".join([f"{word}<{tag}>" for word, tag in zip(sentence.split(), predicted_tags)])
    return result

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
        # Input sentence from user
        sentence = input("Enter a sentence: ").strip()
        
        if sentence.lower() == 'exit':
            logger.info("Exiting inference loop.")
            break
        
        # Log the input sentence
        logger.info(f"Input Sentence: {sentence}")
        
        # Get the POS tagging for the sentence
        result = infer_sentence(model, sentence, word_to_ix, tag_to_ix, config.cuda)
        
        # Log the output sentence
        logger.info(f"Output Tagged Sentence: {result}")
        
        # Display the result
        print(result)

# Entry point of the script
if __name__ == "__main__":
    run_inferencing_loop()
