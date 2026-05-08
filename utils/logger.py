import json
from typing import Any

class Logger:
    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def log(self, epoch: int, loss: float, perplexity: float) -> None:
        self.history.append({"epoch": epoch, "loss": loss, "perplexity": perplexity})
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.history, f)
