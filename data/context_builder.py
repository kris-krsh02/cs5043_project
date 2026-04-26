from sentence_transformers import SentenceTransformer
import torch


class ContextBuilder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: torch.device = None,
    ) -> None:
        self.model = SentenceTransformer(model_name).to(device)
        self.device = device
        self.prompt_embedding: torch.Tensor = None
        self.historic_context: torch.Tensor = torch.zeros(
            self.model.get_sentence_embedding_dimension()
        )
        self.historic_context_embedding: torch.Tensor = torch.zeros(
            self.model.get_sentence_embedding_dimension()
        )

    def build_prompt_embedding(self, prompt: str) -> None:
        self.prompt_embedding = torch.tensor(
            self.model.encode(prompt), dtype=torch.float32
        )

    def update_historic_context(self, new_history) -> None:
        new_history_embedding = torch.tensor(
            self.model.encode(new_history), dtype=torch.float32
        )
        self.historic_context = self.historic_context + new_history_embedding
        norm = torch.linalg.norm(self.historic_context)
        if norm > 0:
            self.historic_context_embedding = self.historic_context / norm
        else:
            self.historic_context_embedding = self.historic_context

    def get_prompt_embedding(self) -> torch.Tensor:
        return self.prompt_embedding

    def get_historic_context_embedding(self) -> torch.Tensor:
        return self.historic_context_embedding

    def reset_history(self) -> None:
        self.historic_context = torch.zeros(
            self.model.get_sentence_embedding_dimension()
        )
        self.historic_context_embedding = torch.zeros(
            self.model.get_sentence_embedding_dimension()
        )
        self.prompt_embedding = None
