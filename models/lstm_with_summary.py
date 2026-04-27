from typing import Optional, Tuple
from models.lstm import LSTMModel
import torch
import torch.nn as nn


class PromptLSTMModel(LSTMModel):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        prompt_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        super(PromptLSTMModel, self).__init__(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout,
            device,
        )
        self.model_type = "prompt"
        self.prompt_dim: int = prompt_dim
        self.lstm = nn.LSTM(
            embedding_dim + prompt_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Implement the forward pass with prompt embedding
        return super().forward(input, state, context)

    def build_input(
        self, input: torch.Tensor, context: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        embedded_input = self.embedding(input)
        seq_len = input.size(1)
        prompt_embedding = context[0]
        prompt_embedding = prompt_embedding.to(self.device)

        # Expand prompt_embedding to match the sequence length
        prompt_embedding = prompt_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat((embedded_input, prompt_embedding), dim=2)
        return lstm_input


class PromptSummaryLSTMModel(LSTMModel):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        prompt_dim: int,
        historic_context_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        super(PromptSummaryLSTMModel, self).__init__(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout,
            device,
        )
        self.model_type = "prompt_summary"
        self.prompt_dim: int = prompt_dim
        self.historic_context_dim: int = historic_context_dim
        self.lstm = nn.LSTM(
            embedding_dim + prompt_dim + historic_context_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Implement the forward pass with prompt embedding
        return super().forward(input, state, context)

    def build_input(
        self, input: torch.Tensor, context: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        embedded_input = self.embedding(input)
        seq_len = input.size(1)
        prompt_embedding = context[0]
        historic_context_embedding = context[1]
        prompt_embedding = prompt_embedding.to(self.device)
        historic_context_embedding = historic_context_embedding.to(self.device)

        # Expand prompt_embedding to match the sequence length
        prompt_embedding = prompt_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        historic_context_embedding = historic_context_embedding.unsqueeze(1).expand(
            -1, seq_len, -1
        )
        lstm_input = torch.cat(
            (embedded_input, prompt_embedding, historic_context_embedding), dim=2
        )
        return lstm_input
