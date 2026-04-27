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
                batch = self.data[i : i + self.config.batch_size].to(self.device)

                if batch.size(0) != self.config.batch_size:
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

                total_seq_len = batch.size(1)
                state = self.model.init_state(self.config.batch_size)
                
                if has_context:
                    for b in range(self.config.batch_size):
                        prompt_tokens = batch[b, : self.config.sequence_length]
                        prompt_text = decode_tokens(prompt_tokens, self.vocab)
                        context_builders[b].build_prompt_embedding(prompt_text)

                for j in range(0, total_seq_len - 1, self.config.sequence_length):
                    input_seq = batch[:, j : j + self.config.sequence_length]
                    target_seq = batch[:, j + 1 : j + self.config.sequence_length + 1]
                    
                    if (
                        input_seq.size(1) != self.config.sequence_length
                        or target_seq.size(1) != self.config.sequence_length
                    ):
                        continue
                    
                    state = self.model.detach_state(state)
                    
                    if has_context:
                        prompt_batch = torch.stack(
                            [cb.get_prompt_embedding() for cb in context_builders]).to(self.device)
                        
                        history_batch = torch.stack(
                            [cb.get_historic_context_embedding() for cb in context_builders]).to(self.device)
                    
                        context = (prompt_batch, history_batch)
                    else:
                        context = None
                    output, state = self.model(input_seq, state, context)

                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)), target_seq.reshape(-1)
                    )
                    
                    perplexity = torch.exp(loss)
                    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if has_context:
                        predictions = get_predicted_tokens(output)
                        for b in range(self.config.batch_size):
                            text = decode_tokens(predictions[b], self.vocab)
                            context_builders[b].update_historic_context(text)

                    self.logger.log(epoch, loss.item(), perplexity.item())

        self.logger.save(f"{self.model.model_type}_training_log.json")
