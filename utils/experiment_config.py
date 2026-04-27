import torch

class ExperimentConfig:
    def __init__(self) -> None:
        self.dataset_name: str = "wikitext"
        self.dataset_specification: str = "wikitext-2-raw-v1"
        self.batch_size: int = 32
        self.embedding_dim: int = 128
        self.hidden_dim: int = 256
        self.num_layers: int = 2
        self.sequence_length: int = 32
        self.dropout: float = 0.5
        self.learning_rate: float = 0.0001
        self.history_window_size: int = 5
        self.num_epochs: int = 30
        self.device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  