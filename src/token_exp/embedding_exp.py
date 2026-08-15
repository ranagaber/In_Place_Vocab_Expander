from transformers import AutoTokenizer , AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

def adjust_model(model_id: str , type : str , tokenizer , new_tokens):
    old_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if type == 'Seq2Seq':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id , dtype = torch.bfloat16)
    elif type == 'LM':    
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype = torch.bfloat16)
    else:
        raise ValueError("type must be either 'Seq2Seq' or 'LM'")
    model.resize_token_embeddings(len(tokenizer))

    embeddings = model.get_input_embeddings()
    with torch.no_grad():
        for token in new_tokens:
            old_ids = old_tokenizer.encode(token , add_special_tokens=False)
            if not old_ids:
                continue
            old_embeddings = embeddings.weight[old_ids]
            new_embeddings = old_embeddings.mean(dim = 0)
            new_id = tokenizer.convert_tokens_to_ids(token)
            embeddings.weight[new_id] = new_embeddings
    return model



