from typing import Optional, Tuple
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Implement the forward pass
        hidden, cell = state
        embedded_input = self.build_input(input, context)
        lstm_out, (hidden, cell) = self.lstm(embedded_input, (hidden, cell))
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, (hidden, cell)

    def build_input(
        self, input: torch.Tensor, context: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        return self.embedding(input)

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize hidden and cell states to zeros
        hidden = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size).to(
            self.device
        )
        cell = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size).to(
            self.device
        )
        return (hidden, cell)

    def detach_state(
        self, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Detach hidden and cell states from the computation graph
        hidden, cell = state
        hidden = hidden.detach()
        cell = cell.detach()
        return (hidden, cell)
