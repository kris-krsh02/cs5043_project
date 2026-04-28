from sentence_transformers import SentenceTransformer
from torch import nn
import torch

from utils.decoder import decode_tokens
from utils.experiment_config import ExperimentConfig
from data.context_builder import ContextBuilder


class Generator:
    def __init__(
        self,
        model: nn.Module,
        vocab: object,
        config: ExperimentConfig,
        shared_embedding_model: SentenceTransformer = None,
        tokenizer = None,
    ) -> None:
        self.model: nn.Module = model
        self.vocab: object = vocab
        self.config: ExperimentConfig = config
        self.shared_embedding_model: SentenceTransformer = shared_embedding_model
        self.tokenizer = tokenizer
        self.model.to(self.config.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        has_prompt: bool,
        has_history: bool,
        max_length: int = 200,
        temperature: float = 0.8,
    ) -> str:
        with torch.no_grad():
            context_builder = None
            if has_prompt:
                context_builder = ContextBuilder(
                    history_window_size=self.config.history_window_size,
                    device=self.config.device,
                    embedding_model=self.shared_embedding_model,
                )
                context_builder.build_prompt_embedding(prompt)
            state = self.model.init_state(1)
            
            encoded_prompt = torch.tensor([self.vocab[token] for token in self.tokenizer(prompt)]).unsqueeze(0).to(self.model.device)
            generated_tokens = encoded_prompt.clone()
            history_buffer = []
            
            for _ in range(max_length):
                input_seq = generated_tokens[:, -self.config.sequence_length :] # Get the last sequence_length tokens
                context = None
                
                if has_prompt:
                    prompt_embedding = context_builder.get_prompt_embedding().unsqueeze(0)
                    history_embedding = None
                    if has_history:
                        history_embedding = context_builder.get_historic_context_embedding().unsqueeze(0)
                    context = (prompt_embedding, history_embedding)
                    
                state = self.model.detach_state(state)
                output, state = self.model(input_seq, state, context)
                
                output = output[:, -1, :] / temperature  
                probabilities = torch.softmax(output, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                
                if has_history:
                    history_buffer.append(next_token)
                    if len(history_buffer) > self.config.sequence_length:
                        text = decode_tokens(history_buffer, self.vocab)
                        context_builder.update_historic_context(text)
                        history_buffer = []
                        
            return decode_tokens(generated_tokens.squeeze(), self.vocab)