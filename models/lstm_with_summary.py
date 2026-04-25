from .lstm import LSTMModel
import torch 

class PromptLSTMModel(LSTMModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5, device=None):
        super(PromptLSTMModel, self).__init__(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, device)
    
    def forward(self, input, state, prompt_embedding):
        # Implement the forward pass with prompt embedding
        hidden, cell = state
        embedded_input = self.embedding(input)

        # Concatenate prompt embedding with input embedding
        combined_input = torch.cat((embedded_input, prompt_embedding), dim=-1)