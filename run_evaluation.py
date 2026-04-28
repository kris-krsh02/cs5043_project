from sentence_transformers import SentenceTransformer
from utils.experiment_config import ExperimentConfig
from data.data_processor import DataProcessor
from models.lstm import LSTMModel
from models.lstm_with_summary import PromptLSTMModel, PromptSummaryLSTMModel
from utils.evaluator import Evaluator
from utils.seed import set_seed
import torch.nn as nn
import torch


def run_evaluation(model_name: str, config: ExperimentConfig = ExperimentConfig(), max_batches: int = None) -> None:
    set_seed(42)
    
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
        checkpoint = torch.load(f"{model_name}_model.pth", map_location=config.device)
        model.load_state_dict(checkpoint)
        model.to(config.device)
    elif model_name == "prompt":
        model = PromptLSTMModel(vocab_size, config.embedding_dim, prompt_history_dim, config.hidden_dim, config.num_layers, config.dropout, device=config.device)
        checkpoint = torch.load(f"{model_name}_model.pth", map_location=config.device)
        model.load_state_dict(checkpoint)
        model.to(config.device)
    elif model_name == "prompt_summary":
        model = PromptSummaryLSTMModel(vocab_size, config.embedding_dim, prompt_history_dim, prompt_history_dim, config.hidden_dim, config.num_layers, config.dropout, device=config.device)
        checkpoint = torch.load(f"{model_name}_model.pth", map_location=config.device)
        model.load_state_dict(checkpoint)
        model.to(config.device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Valid options are 'base', 'prompt', 'prompt_summary'.")

    # Initialize optimizer and criterion
    criterion = nn.CrossEntropyLoss(ignore_index=data_processor.get_pad_idx())

    # Initialize evaluator
    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        config=config,
        data=train_data,
        vocab=vocab,
        shared_embedding_model=shared_embedding_model if model_name != "base" else None,
    )
    print(f"Vocab size: {vocab_size}")
    evaluator.evaluate(has_prompt=(model_name == "prompt" or model_name == "prompt_summary"), has_history=(model_name == "prompt_summary"), max_batches=max_batches)