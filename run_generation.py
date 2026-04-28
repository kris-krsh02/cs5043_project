from sentence_transformers import SentenceTransformer
from utils.experiment_config import ExperimentConfig
from data.data_processor import DataProcessor
from models.lstm import LSTMModel
from models.lstm_with_summary import PromptLSTMModel, PromptSummaryLSTMModel
from utils.generator import Generator
from utils.evaluator import distinct_n, ngram_repetition_rate
from utils.seed import set_seed
import torch


def run_generation(model_name: str, config: ExperimentConfig = ExperimentConfig()) -> None:
    set_seed(42)
    
    # Load data
    data_processor = DataProcessor(config.dataset_name, config.dataset_specification)
    data_processor.load_data()
    data_processor.preprocess_data()
    data_processor.prepare_vocabulary()
    vocab = data_processor.get_vocab()
    vocab_size = data_processor.get_vocab_size()
    
    shared_embedding_model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2",
                device=config.device,
            )
    prompt_history_dim = shared_embedding_model.get_sentence_embedding_dimension()
    
    # Initialize model
    if model_name == "base":
        model = LSTMModel(vocab_size, config.embedding_dim, config.hidden_dim, config.num_layers, config.dropout, device=config.device)
        checkpoint = torch.load(f"checkpoints/{model_name}_model.pth", map_location=config.device)
        model.load_state_dict(checkpoint)
        model.to(config.device)
    elif model_name == "prompt":
        model = PromptLSTMModel(vocab_size, config.embedding_dim, prompt_history_dim, config.hidden_dim, config.num_layers, config.dropout, device=config.device)
        checkpoint = torch.load(f"checkpoints/{model_name}_model.pth", map_location=config.device)
        model.load_state_dict(checkpoint)
        model.to(config.device)
    elif model_name == "prompt_summary":
        model = PromptSummaryLSTMModel(vocab_size, config.embedding_dim, prompt_history_dim, prompt_history_dim, config.hidden_dim, config.num_layers, config.dropout, device=config.device)
        checkpoint = torch.load(f"checkpoints/{model_name}_model.pth", map_location=config.device)
        model.load_state_dict(checkpoint)
        model.to(config.device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Valid options are 'base', 'prompt', 'prompt_summary'.")


    # Initialize generator
    generator = Generator(
        model=model,
        vocab=vocab,
        config=config,
        shared_embedding_model=shared_embedding_model if model_name != "base" else None,
        tokenizer=data_processor.tokenizer,
    )
    
    # Generate text
    prompts = ["The Earth is the third planet from the Sun in the Solar System and",
               "Artificial intelligence is an important field of computer science",
               "Spain is a country in southern Europe with a rich history and",
               "Geometry is a field of mathematics that studies the relationships",
               "Energy is a fundamental concept in physics that describes the ability"]
    generations = []
    distinct2s =[]
    distinct3s = []
    repetition_rates2 = []
    repetition_rates3 = []
    
    for prompt in prompts:
        generated_text = generator.generate(
            prompt=prompt,
            has_prompt=(model_name != "base"),
            has_history=(model_name == "prompt_summary"),
            max_length=200,
            temperature=1.0,
        )
        print("Generated Text for Prompt:", prompt)
        print(generated_text)
        generations.append(generated_text)
        distinct2s.append(distinct_n(generated_text, 2))
        distinct3s.append(distinct_n(generated_text, 3))
        repetition_rates2.append(ngram_repetition_rate(generated_text, 2))
        repetition_rates3.append(ngram_repetition_rate(generated_text, 3))
        print(f"Distinct-2: {distinct2s[-1]:.4f}, Distinct-3: {distinct3s[-1]:.4f}, Repetition Rate-2: {repetition_rates2[-1]:.4f}, Repetition Rate-3: {repetition_rates3[-1]:.4f}")
        
    with open(f"generations/{model_name}_generations.txt", "w") as f:  
        for prompt, gen in zip(prompts, generations):
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated Text: {gen}\n\n")
            f.write(f"Distinct-2: {distinct2s[prompts.index(prompt)]:.4f}\n")
            f.write(f"Distinct-3: {distinct3s[prompts.index(prompt)]:.4f}\n")
            f.write(f"Repetition Rate-2: {repetition_rates2[prompts.index(prompt)]:.4f}\n")
            f.write(f"Repetition Rate-3: {repetition_rates3[prompts.index(prompt)]:.4f}\n")
            f.write("-" * 50 + "\n\n")
            