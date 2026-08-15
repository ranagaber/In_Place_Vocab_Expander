from transformers import AutoTokenizer 
import torch

def expand_tokenizer(model_id: str , vocab:dict):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    old_vocab_size = len(tokenizer)

    old_vocab = tokenizer.get_vocab()
    vocab = vocab.keys()
    new_tokens = [
        token for token in vocab if token not in old_vocab
    ]
    tokenizer.add_tokens(new_tokens)
    print("Existing vocab:", old_vocab_size)
    print("New tokens:", len(new_tokens))
    return tokenizer , new_tokens



