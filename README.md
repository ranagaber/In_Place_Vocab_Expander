# Tokenizer Vocabulary Expansion

This project is designed to address the heavy token fragmentation often observed in low-resource languages.

For example, a pretrained tokenizer may tokenize a diacritized Arabic word into many small subtokens, effectively approaching character-level tokenization. We can train a new SentencePiece tokenizer on text from an under-resourced language, add its new vocabulary to the pretrained tokenizer, and initialize the embeddings of the newly added tokens using the pretrained model's existing embeddings. 
The tool has an option to automatically perform Continued Pretraining (CPT) on **ONLY NEW EMBEDDINGS** using the same corpus used for vocabulary expansion to further refine the newly initialized embeddings.
Further CPT for full weights is preferred.
> **CUDA is required for Continued Pretraining.**

The workflow is:

```text
Arabic corpus
     ↓
Train SentencePiece tokenizer
     ↓
Extract vocabulary
     ↓
Add new tokens to pretrained tokenizer
     ↓
Resize model embeddings
     ↓
Initialize new embeddings from the mean of old-token fragments
     ↓
Optional Continued Pretraining (CPT)
     ↓
Refine the newly added embeddings only
```
# Evaluation

The tokenizer was evaluated on a held-out corpus of **3,000 samples from the Tashkeela corpus**. These samples are distinct from the data used to train the tokenizer.

The evaluation compares the original `google/gemma-3-1b-pt` tokenizer with the expanded `RanaGaber/gemma_1B_pt_diac` tokenizer.

| Metric           | `google/gemma-3-1b-pt` | `RanaGaber/gemma_1B_pt_diac` |
| ---------------- | ---------------------: | ---------------------------: |
| **Total Words**  |                244,114 |                      244,114 |
| **Total Tokens** |              1,019,110 |                      343,313 |
| **Fertility**    |                  4.175 |                    **1.406** |


## Project Structure

```text
.
├── token_exp/
│   ├── new_vocab.py
│   ├── tokenizer_exp.py
    ├── configs.py
    ├── cpt.py
    ├── main.py
│   └── embedding_exp.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

```bash
uv sync
```
## Usage

The project provides a command-line interface (CLI) for tokenizer expansion with optional Continued Pretraining (CPT).

### Run without CPT

```bash
uv run main.py \
    --model-id "your/model" \
    --txt-path "path/to/corpus.txt" \
    --type "LM" \
    --token "your_huggingface_token" \
    --repo-id "your/expanded-model-repo" \
    --final-repo "your/cpt-model-repo"

### Run with CPT
Add the --cpt flag:
```bash
uv run main.py \
    --model-id "your/model" \
    --txt-path "path/to/corpus.txt" \
    --type "LM" \
    --token "your_huggingface_token" \
    --repo-id "your/expanded-model-repo" \
    --final-repo "your/cpt-model-repo" \
    --cpt
### Arguments

| Argument     | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| `txt_path`   | Path to the corpus used to train the new SentencePiece tokenizer and perform CPT |
| `model_id`   | Hugging Face model ID of the pretrained model                                    |
| `type`       | `"LM"` for causal language models or `"Seq2Seq"` for encoder-decoder models      |
| `token`      | Hugging Face access token used to upload the model                               |
| `repo_id`    | Hugging Face repository for the expanded model                                   |
| `final_repo` | Hugging Face repository for the final CPT model                                  |
| `cpt`        | Boolean value to signal whether to perform CPT                               |



## Citation

This work is inspired by:

```bibtex
@misc{smith2026inplacetokenizerexpansionpretrained,
      title={In-Place Tokenizer Expansion for Pre-trained LLMs}, 
      author={Jimmy T. H. Smith and Tarek Dakhran and Alberto Cabrera and Simon S. Lee and Paul Pak and Aditya Tadimeti and Tim Seyde and Maxime Labonne and Alexander Amini and Mathias Lechner},
      year={2026},
      eprint={2607.15232},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.15232}, 
}
