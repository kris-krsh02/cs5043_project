from typing import List
import math
import torch
import torch.nn as nn
import random
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
        shared_embedding_model: SentenceTransformer = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.logger = Logger()
        self.data = data
        self.vocab = vocab
        self.device = torch.device(self.config.device)
        self.shared_embedding_model = shared_embedding_model
        self.model.to(self.device)
        self.model.device = self.device

    def train(self, has_prompt: bool, has_history: bool, max_batches: int = None) -> None:
        model_type = self.model.model_type

        if model_type == "base" and (has_prompt or has_history):
            raise ValueError(
                "Invalid config: base model cannot use context. Fix: call train(has_prompt=False, has_history=False) or switch model_type to 'prompt'/'prompt_summary'."
            )

        if model_type == "prompt" and has_prompt and has_history: 
            raise ValueError(
                "Invalid config: prompt model cannot use history context. Fix: call train(has_prompt=True, has_history=False) or switch model_type to 'prompt_summary'."
            )
        elif model_type == "prompt" and not has_prompt:
            raise ValueError(
                "Invalid config: prompt model must use prompt context. Fix: call train(has_prompt=True, has_history=False) or switch model_type to 'base'."
            )
            
        if model_type == "prompt_summary" and not (has_prompt and has_history):
            raise ValueError(
                "Invalid config: prompt_summary model must use both prompt and history context. Fix: call train(has_prompt=True, has_history=True) or switch model_type to 'base'/'prompt'."
            )

        self.model.train()            

        for epoch in range(self.config.num_epochs):
            total_loss = 0.0
            total_tokens = 0
            
            random.seed(epoch + 42)  # Ensure different shuffling each epoch
            random.shuffle(self.data)  # Shuffle data due to batch limit to ensure randomness
            
            for i in range(0, len(self.data), self.config.batch_size):
                if max_batches is not None and i // self.config.batch_size >= max_batches:
                    break
                
                batch_docs = self.data[i : i + self.config.batch_size]
                if len(batch_docs) < self.config.batch_size:
                    continue
                
                context_builders: List[ContextBuilder] = []

                if has_prompt:
                    context_builders = [
                        ContextBuilder(
                            history_window_size=self.config.history_window_size,
                            device=self.device,
                            embedding_model=self.shared_embedding_model,
                        )
                        for _ in range(self.config.batch_size)
                    ]
                    
                    # Build prompts in the beginning
                    for b in range(self.config.batch_size):
                        prompt = decode_tokens(batch_docs[b][0][:-1], self.vocab) # Use first sequence of each doc as prompt
                        context_builders[b].build_prompt_embedding(prompt)
                        
                state = self.model.init_state(self.config.batch_size)
                num_steps = max(len(text) for text in batch_docs) # based on longest article
                
                for t in range(num_steps):
                    inputs, targets = [], []
                    
                    # Handle differently sized documents
                    for doc in batch_docs:
                        if t < len(doc):
                            inputs.append(doc[t][:-1]) 
                            targets.append(doc[t][1:])  
                        else:
                            inputs.append(torch.full((self.config.sequence_length,), self.vocab["<pad>"], dtype=torch.long))
                            targets.append(torch.full((self.config.sequence_length,), self.vocab["<pad>"], dtype=torch.long))
                    
                    input_seq = torch.stack(inputs).to(self.device)
                    target_seq = torch.stack(targets).to(self.device)
                    
                    if has_prompt:
                        prompt_batch = torch.stack(
                            [cb.get_prompt_embedding() for cb in context_builders]).to(self.device)
                        
                        history_batch = None
                        if has_history:
                            history_batch = torch.stack(
                                [cb.get_historic_context_embedding() for cb in context_builders]).to(self.device)
                    
                        context = (prompt_batch, history_batch)
                    else:
                        context = None
                    
                    state = self.model.detach_state(state)
                    output, state = self.model(input_seq, state, context)


                    # Compute number of valid (non-pad) tokens in this step and skip if none
                    pad_idx = self.vocab["<pad>"]
                    num_valid_tokens = (target_seq.reshape(-1) != pad_idx).sum().item()
                    if num_valid_tokens == 0:
                        # nothing to learn from this time-step (all pads)
                        continue

                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)), target_seq.reshape(-1)
                    )

                    # Compute number of valid (non-pad) tokens in this step
                    pad_idx = self.vocab["<pad>"]
                    num_valid_tokens = (target_seq.reshape(-1) != pad_idx).sum().item()

                    # loss is mean over non-ignored tokens (CrossEntropyLoss with ignore_index),
                    # so multiply by num_valid_tokens to get token-sum, then accumulate
                    total_loss += loss.item() * num_valid_tokens
                    total_tokens += num_valid_tokens

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                    self.optimizer.step()
                
                    if has_history:
                        for b in range(self.config.batch_size):
                            text = decode_tokens(target_seq[b], self.vocab)
                            context_builders[b].update_historic_context(text)

            epoch_avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
            epoch_perplexity = math.exp(epoch_avg_loss) if math.isfinite(epoch_avg_loss) else float("nan")

            self.logger.log(epoch, epoch_avg_loss, epoch_perplexity)

        self.logger.save(f"{self.model.model_type}_training_log.json")
