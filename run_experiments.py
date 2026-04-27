from sentence_transformers import SentenceTransformer

from utils.experiment_config import ExperimentConfig
from data.data_processor import DataProcessor
from models.lstm import LSTMModel
from models.lstm_with_summary import PromptLSTMModel, PromptSummaryLSTMModel
from utils.trainer import Trainer
import torch
import torch.nn as nn


def run_experiment(model_name: str, config: ExperimentConfig = ExperimentConfig(), max_batches: int = None) -> None:
    # Load data
    data_processor = DataProcessor(config.dataset_name, config.dataset_specification)
    data_processor.load_data()
    data_processor.preprocess_data()
    data_processor.prepare_vocabulary()
    vocab = data_processor.get_vocab()
    train_data = data_processor.get_data("train", config.sequence_length)
    vocab_size = data_processor.get_vocab_size()
    
    shared_embedding_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2",
                device=config.device,
            )
    prompt_history_dim = shared_embedding_model.get_sentence_embedding_dimension()
    
    # Initialize model
    if model_name == "base":
        model = LSTMModel(vocab_size, config.embedding_dim, config.hidden_dim, config.num_layers, config.dropout, device=config.device)
    elif model_name == "prompt":
        model = PromptLSTMModel(vocab_size, config.embedding_dim, prompt_history_dim, config.hidden_dim, config.num_layers, config.dropout, device=config.device)
    elif model_name == "prompt_summary":
        model = PromptSummaryLSTMModel(vocab_size, config.embedding_dim, prompt_history_dim, prompt_history_dim, config.hidden_dim, config.num_layers, config.dropout, device=config.device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Valid options are 'base', 'prompt', 'prompt_summary'.")

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=data_processor.get_pad_idx())

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        data=train_data,
        vocab=vocab,
        shared_embedding_model=shared_embedding_model if model_name != "base" else None,
    )
    print(f"Vocab size: {vocab_size}")
    trainer.train(has_prompt=(model_name == "prompt" or model_name == "prompt_summary"), has_history=(model_name == "prompt_summary"), max_batches=max_batches)