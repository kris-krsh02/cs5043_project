from typing import List, Tuple
import math
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from utils.decoder import decode_tokens, get_predicted_tokens
from utils.logger import Logger
from data.context_builder import ContextBuilder


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config,
        data: torch.Tensor,
        vocab: object,
        shared_embedding_model: SentenceTransformer = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.config = config
        self.logger = Logger()
        self.data = data
        self.vocab = vocab
        self.device = torch.device(self.config.device)
        self.shared_embedding_model = shared_embedding_model
        self.model.to(self.device)
        self.model.device = self.device

    def evaluate(self, has_prompt: bool, has_history: bool, max_batches: int = None) -> None:
        model_type = self.model.model_type
        model_types = ["base", "prompt", "prompt_summary"]

        if model_type == "base" and (has_prompt or has_history):
            raise ValueError(
                "Invalid config: base model cannot use context. Fix: call evaluate(has_prompt=False, has_history=False) or switch model_type to 'prompt'/'prompt_summary'."
            )

        if model_type == "prompt" and has_prompt and has_history: 
            raise ValueError(
                "Invalid config: prompt model cannot use history context. Fix: call evaluate(has_prompt=True, has_history=False) or switch model_type to 'prompt_summary'."
            )
        elif model_type == "prompt" and not has_prompt:
            raise ValueError(
                "Invalid config: prompt model must use prompt context. Fix: call evaluate(has_prompt=True, has_history=False) or switch model_type to 'base'."
            )
            
        if model_type == "prompt_summary" and not (has_prompt and has_history):
            raise ValueError(
                "Invalid config: prompt_summary model must use both prompt and history context. Fix: call evaluate(has_prompt=True, has_history=True) or switch model_type to 'base'/'prompt'."
            )

        self.model.eval()            

        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(self.data), self.config.batch_size):
                print(f"Evaluating batch {i // self.config.batch_size + 1}")
                if max_batches is not None and i // self.config.batch_size >= max_batches:
                    break
                
                batch_seq = self.data[i : i + self.config.batch_size]
                if len(batch_seq) < self.config.batch_size:
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
                    
                    if has_prompt and t == 0:
                        for b in range(self.config.batch_size):
                            prompt_tokens = batch[b, : -1]
                            prompt_text = decode_tokens(prompt_tokens, self.vocab)
                            context_builders[b].build_prompt_embedding(prompt_text)
                
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
                    
                    num_tokens = target_seq.numel()
                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)), target_seq.reshape(-1)
                    )
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
                    perplexity = torch.exp(loss)
                
                    if has_history:
                        predictions = get_predicted_tokens(output)
                        for b in range(self.config.batch_size):
                            text = decode_tokens(predictions[b], self.vocab)
                            context_builders[b].update_historic_context(text)
                    
                    self.logger.log(i // self.config.batch_size, loss.item(), perplexity.item())

            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")

            perplexity = math.exp(avg_loss) if math.isfinite(avg_loss) else float("nan")
            self.logger.log(model_types.index(self.model.model_type), avg_loss, perplexity)
            print(f"Evaluation complete for {self.model.model_type}. Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

        self.logger.save("evaluation_log.json")
