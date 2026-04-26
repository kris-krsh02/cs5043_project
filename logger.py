import json


class Logger:
    def __init__(self):
        self.history = []

    def log(self, epoch: int, loss: float) -> None:
        self.history.append((epoch, loss))
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.history, f)
