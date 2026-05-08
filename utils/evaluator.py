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
                num_steps = max(len(text) for text in batch_seq)

                # If using prompts, build them once from the first sequence of each document
                if has_prompt:
                    for b in range(self.config.batch_size):
                        prompt = decode_tokens(batch_seq[b][0][:-1], self.vocab)
                        context_builders[b].build_prompt_embedding(prompt)
                        
                history_buffers = [[] for _ in range(self.config.batch_size)]

                for t in range(num_steps):
                    inputs, targets = [], []
                    for doc in batch_seq:
                        if t < len(doc):
                            inputs.append(doc[t][:-1])
                            targets.append(doc[t][1:])
                        else:
                            inputs.append(
                                torch.full((self.config.sequence_length,), self.vocab["<pad>"], dtype=torch.long)
                            )
                            targets.append(
                                torch.full((self.config.sequence_length,), self.vocab["<pad>"], dtype=torch.long)
                            )

                    input_seq = torch.stack(inputs).to(self.device)
                    target_seq = torch.stack(targets).to(self.device)

                    if has_prompt:
                        prompt_batch = torch.stack(
                            [cb.get_prompt_embedding() for cb in context_builders]
                        ).to(self.device)

                        history_batch = None
                        if has_history:
                            history_batch = torch.stack(
                                [cb.get_historic_context_embedding() for cb in context_builders]
                            ).to(self.device)

                        context = (prompt_batch, history_batch)
                    else:
                        context = None

                    state = self.model.detach_state(state)
                    output, state = self.model(input_seq, state, context)

                    # Skip if all targets are padding for this step
                    pad_idx = self.vocab["<pad>"]
                    num_valid_tokens = (target_seq.reshape(-1) != pad_idx).sum().item()
                    if num_valid_tokens == 0:
                        continue

                    loss = self.criterion(
                        output.reshape(-1, output.size(-1)), target_seq.reshape(-1)
                    )
                    perplexity = torch.exp(loss)

                    total_loss += loss.item() * num_valid_tokens
                    total_tokens += num_valid_tokens

                    # Update history from predictions, matching Generator's token-by-token accumulation
                    if has_history:
                        predictions = get_predicted_tokens(output)
                        for b in range(self.config.batch_size):
                            next_token = predictions[b, -1]
                            history_buffers[b].append(next_token)

                            if len(history_buffers[b]) == self.config.sequence_length:
                                text = decode_tokens(history_buffers[b], self.vocab)
                                context_builders[b].update_historic_context(text)
                                history_buffers[b] = []
    
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
            perplexity = math.exp(avg_loss) if math.isfinite(avg_loss) else float("nan")
            self.logger.log(model_types.index(self.model.model_type), avg_loss, perplexity)
            print(f"Evaluation complete for {self.model.model_type}. Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

        self.logger.save(f"logs/{self.model.model_type}_evaluation_log.json")


def distinct_n(text: str, n: int) -> float:
    "Calculates the proportion of unique n-grams in the given text."
    tokens = text.split()
    total_ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    distinct_ngrams = set(total_ngrams)
    return len(distinct_ngrams) / len(total_ngrams) if total_ngrams else 0.0

def ngram_repetition_rate(text: str, n: int) -> float:
    "Calculates the repetition rate of n-grams in the given text."
    tokens = text.split()
    total_ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    ngram_counts = {}
    for ngram in total_ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
    return repeated_ngrams / len(total_ngrams) if total_ngrams else 0.0