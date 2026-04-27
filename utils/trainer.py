from typing import List
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from utils.decoder import decode_tokens, get_predicted_tokens
from utils.logger import Logger
from data.context_builder import ContextBuilder


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config,
        data: torch.Tensor,
        vocab: object,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.logger = Logger()
        self.data = data
        self.vocab = vocab
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.model.device = self.device

    def train(self, has_context: bool, max_batches: int = None) -> None:
        model_type = self.model.model_type

        if model_type == "base" and has_context:
            raise ValueError(
                "Invalid config: base model cannot use context. Fix: call train(has_context=False) or switch model_type to 'prompt'/'prompt_summary'."
            )

        if model_type in {"prompt", "prompt_summary"} and not has_context:
            raise ValueError(
                "Invalid config: prompt/context models require context. Fix: call train(has_context=True) or switch model_type to 'base'."
            )

        self.model.train()

        shared_embedding_model = None
        if has_context:
            shared_embedding_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2",
                device=self.device,
            )

        for epoch in range(self.config.num_epochs):
            
            for i in range(0, len(self.data), self.config.batch_size):
                if max_batches is not None and i // self.config.batch_size >= max_batches:
                    break
                
                batch_seq = self.data[i : i + self.config.batch_size]
                if len(batch_seq) < self.config.batch_size:
                    continue
                
                
                if has_context:
                    context_builders: List[ContextBuilder] = [
                        ContextBuilder(
                            history_window_size=self.config.history_window_size,
                            device=self.device,
                            embedding_model=shared_embedding_model,
                        )
                        for _ in range(self.config.batch_size)
                    ]
                        
                state = self.model.init_state(self.config.batch_size)
                num_steps = min(len(text) for text in batch_seq)
                
                for t in range(num_steps):
                    batch = torch.stack([text[t] for text in batch_seq]).to(self.device)
                    input_seq = batch[:, :-1]
                    target_seq = batch[:, 1:]

                    if (
                        input_seq.size(1) != self.config.sequence_length
                        or target_seq.size(1) != self.config.sequence_length
                    ):
                        continue
                    
                    if has_context and t == 0:
                        for b in range(self.config.batch_size):
                            prompt_tokens = batch[b, : -1]
                            prompt_text = decode_tokens(prompt_tokens, self.vocab)
                            context_builders[b].build_prompt_embedding(prompt_text)
                
                    if has_context:
                        prompt_batch = torch.stack(
                            [cb.get_prompt_embedding() for cb in context_builders]).to(self.device)
                        
                        history_batch = torch.stack(
                            [cb.get_historic_context_embedding() for cb in context_builders]).to(self.device)
                    
                        context = (prompt_batch, history_batch)
                    else:
                        context = None
                    
                    state = self.model.detach_state(state)
                    output, state = self.model(input_seq, state, context)

                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)), target_seq.reshape(-1)
                    )
                    perplexity = torch.exp(loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                    self.optimizer.step()
                
                    if has_context:
                        predictions = get_predicted_tokens(output)
                        for b in range(self.config.batch_size):
                            text = decode_tokens(predictions[b], self.vocab)
                            context_builders[b].update_historic_context(text)

                    self.logger.log(epoch, loss.item(), perplexity.item())

        self.logger.save(f"{self.model.model_type}_training_log.json")
