import torch


def get_predicted_tokens(output: torch.Tensor) -> torch.Tensor:
    predicted_tokens = torch.argmax(output, dim=-1)
    return predicted_tokens

def decode_tokens(tokens: torch.Tensor, vocab):
    decoded_tokens = []
    
    for token in tokens:
        decoded_tokens.append(vocab.lookup_token(token.item()))
        
    return " ".join(decoded_tokens)
