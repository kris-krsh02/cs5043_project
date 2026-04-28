from typing import Optional, List
import datasets
import torchtext
import torch
from torch.nn.utils.rnn import pad_sequence


class DataProcessor:
    def __init__(self, dataset_name: str, data_specification: str) -> None:
        self.dataset_name: str = dataset_name
        self.data_specification: str = data_specification
        self.dataset: datasets.DatasetDict = None
        self.tokenized_dataset: datasets.DatasetDict = datasets.DatasetDict()
        self.vocab: Optional[object] = None
        self.tokenizer = None

    def load_data(self) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name, self.data_specification)
        print(f"Loaded dataset: \n {self.dataset}")

    def preprocess_data(self) -> None:
        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.tokenizer = tokenizer
        
        # Tokenize whole dataset
        for split in self.dataset.keys():
            self.tokenized_dataset[split] = self.dataset[split].map(
                lambda x: {"tokens": tokenizer(x["text"])}
            )

    def prepare_vocabulary(self) -> None:
        specials = ["<pad>", "<unk>", "<eos>"]

        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            self.tokenized_dataset["train"]["tokens"],
            min_freq=3,
            specials=specials,
            special_first=True,
        )
        self.vocab.set_default_index(self.vocab["<unk>"])

        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Sample tokens: {self.vocab.get_itos()[:10]}")

    def get_data(self, split: str, sequence_length: int) -> torch.Tensor:
        data: List[List[torch.Tensor]] = []
        
        for entry in self.tokenized_dataset[split]:
            tokens = entry["tokens"] + ["<eos>"]
            tokens = [self.vocab[token] for token in tokens]
            
            sequences = []
            for i in range(0, len(tokens) - sequence_length):
                seq = tokens[i : i + sequence_length + 1]
                sequences.append(torch.tensor(seq, dtype=torch.long))

            if len(sequences) > 0:
                data.append(sequences)
        return data
        
    
    def get_vocab(self) -> Optional[object]:
        return self.vocab
    
    def get_vocab_size(self) -> int:
        if self.vocab is None:
            raise ValueError("Vocabulary not prepared yet. Call prepare_vocabulary() first.")
        return len(self.vocab)

    def get_pad_idx(self) -> int:
        return self.vocab["<pad>"]


# Code reference: https://medium.com/data-science/language-modeling-with-lstms-in-pytorch-381a26badcbf