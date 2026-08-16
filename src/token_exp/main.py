from token_exp.new_vocab import pass_vocab
from token_exp.tokenizer_exp import expand_tokenizer
from token_exp.embedding_exp import adjust_model
from huggingface_hub import upload_folder , login, create_repo
from token_exp.cpt import run_cpt
from token_exp.configs import *
from datasets import load_dataset , load_from_disk , Dataset
from transformers import set_seed
from token_exp.freeze_and_unfreeze import *
import random 
import numpy as np
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Expand Your Tokenizer and perform CPT!'
    )
    parser.add_argument(
        "--model-id", 
        type = str,
        required = True, 
        help = 'HuggingFace Model ID'
    )
    parser.add_argument(
        "--txt-path",
        type = str,
        required = True,
        help = 'Path to the corpus'
    )
    parser.add_argument(
        "--type",
        type = str,
        required = True,
        help = "LM or Seq2Seq"
    )
    parser.add_argument(
        "--token",
        type = str, 
        required = True, 
        help = "Hugging Face access token used to upload the model"
    )
    parser.add_argument(
        "--repo-id",
        type = str, 
        required = True, 
        help = "Hugging Face repository for the expanded model"
    )
    parser.add_argument(
        "--final-repo",
        type = str, 
        required = True, 
        help = "Hugging Face repository for the final CPT model"
    )
    parser.add_argument(
        "--cpt",
        action="store_true",
        help="Perform CPT after vocabulary expansion."
    )
    return parser.parse_args()


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
    tokenizer , new_tokens , new_token_ids = expand_tokenizer(model_id , vocab) 
    model = adjust_model(model_id, type, tokenizer, new_tokens)
    return model, tokenizer , new_token_ids
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
    args = parse_args()
    print("In-Place Tokenizer Expansion")
    print("=" * 40)
    print(f"Model : {args.model_id}")
    print(f"Corpus: {args.txt_path}")
    print(f"Type  : {args.type}")
    print(f"CPT   : {args.cpt}")
    print("=" * 40)

    token = args.token
    repo_id = args.repo_id
    model_id = args.model_id
    txt_path = args.txt_path
    type = args.type #LM or Seq2Seq
    final_repo = args.final_repo
    cpt = args.cpt
    



    
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        sentences = [line.strip() for line in f if line.strip()]
    dataset = Dataset.from_dict({
              "Text": sentences})
    model, tokenizer , new_token_ids= main(
        txt_path=txt_path,
        model_id=model_id,
        type=type
    )
    save(model , tokenizer)
    upload(model, tokenizer, token = token , repo_id = repo_id)
    if cpt:
        if tokenizer.pad_token is None:
           tokenizer.pad_token = tokenizer.eos_token
        freeze_all_except_new_embeddings(model , new_token_ids)
        new_model , new_tokenizer = run_cpt(
                               model=model,
                               train_dataset= dataset,
                               output_dir=OUTPUT_DIR,
                               peak_lr = lr,
                               warmup_ratio=WARMUP_RATIO,
                               epochs=NUM_EPOCHS,
                               tokenizer=tokenizer,)
        unfreeze_all(new_model)
        upload(new_model, new_tokenizer, token = token , repo_id = final_repo)


