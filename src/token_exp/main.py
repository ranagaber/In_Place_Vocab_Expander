from token_exp.new_vocab import pass_vocab
from token_exp.tokenizer_exp import expand_tokenizer
from token_exp.embedding_exp import adjust_model
from huggingface_hub import upload_folder , login, create_repo
from token_exp.cpt import run_cpt
from token_exp.configs import *
from datasets import load_dataset , load_from_disk , Dataset
from transformers import set_seed
import random 
import numpy as np
import torch


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(txt_path : str, model_id: str , type : str):
    vocab = pass_vocab(txt_path)
    tokenizer , new_tokens = expand_tokenizer(model_id , vocab) 
    model = adjust_model(model_id, type, tokenizer, new_tokens)
    return model, tokenizer
def save(model , tokenizer , model_dir="new_model_and_tokenizer"):
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

def upload(model ,  tokenizer , token : str , repo_id : str , model_dir = "new_model_and_tokenizer"):
    login(token = token)
    create_repo(
        repo_id = repo_id,
        exist_ok = True,
        token = token,
        repo_type = 'model'
    )
    upload_folder(
        repo_id = repo_id, 
        folder_path = model_dir,
        repo_type = 'model'

    )

if __name__ == "__main__": 
    token = "your token"
    repo_id = "your id"
    model_id = "your model"
    txt_path = "your txt path"
    type = "" #LM or Seq2Seq
    final_repo = "your repo id after cpt"
    
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        sentences = [line.strip() for line in f if line.strip()]
    dataset = Dataset.from_dict({
              "Text": sentences})
    model, tokenizer = main(
        txt_path=txt_path,
        model_id=model_id,
        type=type
    )
    save(model , tokenizer)
    upload(model, tokenizer, token = token , repo_id = repo_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    new_model , new_tokenizer = run_cpt(
        model=model,
        train_dataset= dataset,
        output_dir=OUTPUT_DIR,
        peak_lr = lr,
        warmup_ratio=WARMUP_RATIO,
        epochs=NUM_EPOCHS,
        tokenizer=tokenizer,
    )
    upload(new_model, new_tokenizer, token = token , repo_id = final_repo)

