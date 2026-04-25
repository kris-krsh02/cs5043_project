import torch 
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5, device=None):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        

    def forward(self, input, state):
        # Implement the forward pass
        hidden, cell = state
        embedded_input = self.embedding(input)
        lstm_out, (hidden, cell) = self.lstm(embedded_input, (hidden, cell))
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, (hidden, cell)
    
    def init_state(self, batch_size):
        # Initialize hidden and cell states to zeros
        hidden = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        cell = torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        return (hidden, cell)
    
    def detach_state(self, state):
        # Detach hidden and cell states from the computation graph
        hidden, cell = state
        hidden = hidden.detach()
        cell = cell.detach()
        return (hidden, cell)
    
    