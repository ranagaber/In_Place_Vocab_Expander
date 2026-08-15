from token_exp.new_vocab import pass_vocab
from token_exp.tokenizer_exp import expand_tokenizer
from token_exp.embedding_exp import adjust_model
def main(txt_path : str, model_id: str , type : str):
    vocab = pass_vocab(txt_path)
    tokenizer , new_tokens = expand_tokenizer(model_id , vocab) 
    model = adjust_model(model_id, type, tokenizer, new_tokens)
    return model, tokenizer
def save(model , tokenizer , model_dir="new_model", tokenizer_dir="expanded_tokenizer"):
    tokenizer.save_pretrained(tokenizer_dir)
    model.save_pretrained(model_dir)
    
if __name__ == "__main__":
    model, tokenizer = main(
        txt_path="your/path",
        model_id="your/model",
        type="LM"
    )
    save(model , tokenizer)
