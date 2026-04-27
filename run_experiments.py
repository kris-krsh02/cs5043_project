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
    train_data = data_processor.get_data("train")
    
    # Initialize model
    if model_name == "base":
        model = LSTMModel(config.vocab_size, config.embedding_dim, config.hidden_dim)
    elif model_name == "prompt":
        model = PromptLSTMModel(config.vocab_size, config.embedding_dim, config.hidden_dim)
    elif model_name == "prompt_summary":
        model = PromptSummaryLSTMModel(config.vocab_size, config.embedding_dim, config.hidden_dim)
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
        vocab=vocab
    )
    
    trainer.train(has_context=(model_name != "base"), max_batches=max_batches)