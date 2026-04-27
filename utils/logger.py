import json


class Logger:
    def __init__(self):
        self.history = []

    def log(self, epoch: int, loss: float, perplexity: float) -> None:
        self.history.append((epoch, loss, perplexity))
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.history, f)
