import sentencepiece as spm

def pass_vocab(txt_path: str , vocab_size: int = 30000):
    prefix = "arabic"
    spm.SentencePieceTrainer.train(
    input=txt_path,
    model_prefix=prefix,
    vocab_size=vocab_size,)
    sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")
    vocab = {
        sp.id_to_piece(i) : i 
        for i in range(sp.get_piece_size())
        if sp.id_to_piece(i) not in {"<unk>", "<s>", "</s>"}

    }
    return vocab