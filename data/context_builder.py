from sentence_transformers import SentenceTransformer
import torch


class ContextBuilder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        history_window_size: int = 5,
        device: torch.device = None,
        embedding_model: SentenceTransformer = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = (
            embedding_model
            if embedding_model is not None
            else SentenceTransformer(model_name, device=self.device)
        )
        self.new_history: str = ""
        self.prompt_embedding: torch.Tensor = None
        self.historic_context: torch.Tensor = torch.zeros(
            self.model.get_sentence_embedding_dimension(), device=self.device
        )
        self.historic_context_embedding: torch.Tensor = torch.zeros(
            self.model.get_sentence_embedding_dimension(), device=self.device
        )
        self.time_step: int = 0
        self.history_window_size: int = history_window_size

    def build_prompt_embedding(self, prompt: str) -> None:
        self.prompt_embedding = torch.tensor(
            self.model.encode(prompt), dtype=torch.float32, device=self.device
        )

    def update_historic_context(self, new_history: str) -> None:
        self.new_history += " " + new_history
        self.time_step += 1
        
        if self.time_step < self.history_window_size:
            return
        
        new_history_embedding = torch.tensor(
            self.model.encode(self.new_history), dtype=torch.float32, device=self.device
        )
        self.historic_context = self.historic_context + new_history_embedding
        norm = torch.linalg.norm(self.historic_context)
        if norm > 0:
            self.historic_context_embedding = self.historic_context / norm
        else:
            self.historic_context_embedding = self.historic_context
            
        self.new_history = ""
        self.time_step = 0

    def get_prompt_embedding(self) -> torch.Tensor:
        if self.prompt_embedding is None:
            return torch.zeros(self.model.get_sentence_embedding_dimension(), device=self.device)
        return self.prompt_embedding

    def get_historic_context_embedding(self) -> torch.Tensor:
        return self.historic_context_embedding

    def reset_history(self) -> None:
        self.historic_context = torch.zeros(
            self.model.get_sentence_embedding_dimension(), device=self.device
        )
        self.historic_context_embedding = torch.zeros(
            self.model.get_sentence_embedding_dimension(), device=self.device
        )
        self.prompt_embedding = None
        self.new_history = ""
        self.time_step = 0
