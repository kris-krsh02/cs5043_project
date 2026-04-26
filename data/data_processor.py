from typing import Optional, List
import datasets
import torchtext
import torch


class DataProcessor:
    def __init__(self, dataset_name: str, data_specification: str) -> None:
        self.dataset_name: str = dataset_name
        self.data_specification: str = data_specification
        self.dataset: Optional[datasets.DatasetDict] = None
        self.tokenized_dataset: Optional[datasets.DatasetDict] = None
        self.vocab: Optional[object] = None

    def load_data(self) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name, self.data_specification)
        print(f"Loaded dataset: \n {self.dataset}")

    def preprocess_data(self) -> None:
        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

        # Tokenize whole dataset
        for split in self.dataset.keys():
            for i in range(len(self.dataset[split])):
                self.tokenized_dataset[split][i]["tokens"] = tokenizer(
                    self.dataset[split][i]["text"]
                )

    def prepare_vocabulary(self) -> None:
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            self.tokenized_dataset["train"]["tokens"], min_freq=3
        )
        self.vocab.insert_token("<unk>", 0)
        self.vocab.insert_token("<eos>", 1)
        self.vocab.set_default_index(self.vocab["<unk>"])

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Sample tokens: {self.vocab.get_itos()[:10]}")

    def get_data(self, split: str, batch_size: int) -> torch.Tensor:
        data: List[List[int]] = []
        for entry in self.tokenized_dataset[split]:
            tokens = entry["tokens"].append("<eos>")
            tokens = [self.vocab[token] for token in tokens]
            data.append(tokens)

        data_tensor = torch.tensor(data)
        num_batches = data_tensor.shape[0] // batch_size
        data_tensor = data_tensor[: num_batches * batch_size].view(
            batch_size, num_batches
        )
        return data_tensor


# Code reference: https://medium.com/data-science/language-modeling-with-lstms-in-pytorch-381a26badcbf