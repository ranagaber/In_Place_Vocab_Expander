from transformers import Trainer, TrainingArguments
import torch

def freeze_all_except_new_embeddings(model, new_token_ids):
    for param in model.parameters():
        param.requires_grad = False

    embeddings = model.get_input_embeddings()
    embeddings.weight.requires_grad = True

    new_ids = list(new_token_ids.values())

    def hook(grad):
        grad[:min(new_ids)] = 0
        return grad
    embeddings.weight.register_hook(hook)
    
def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
