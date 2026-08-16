#MODEL AND TRAINING HYPERPARAMETERS
lr = 3e-5
TRAIN_BATCH = 8
WARMUP_RATIO = 0.03
VAL_BATCH = 4
GRAD_ACCUM = 8
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01
SAVE_STEPS = 100
EVAL_STEPS = 1000
LOGGING_STEPS = 100
OUTPUT_DIR = "./outputs"


# Tokenizer Vocabulary Expansion

This project is designed to address the heavy token fragmentation often observed in low-resource languages.

For example, a pretrained tokenizer may tokenize a diacritized Arabic word into many small subtokens, effectively approaching character-level tokenization. We train a new SentencePiece tokenizer on text from an under-resourced language, add its vocabulary to the pretrained tokenizer, and initialize the embeddings of the newly added tokens using the pretrained model's existing embeddings.

The tool then **automatically performs Continued Pretraining (CPT)** using the same corpus used for vocabulary expansion to further refine the newly initialized embeddings.

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
Continued Pretraining (CPT)
     ↓
Refine the newly added embeddings
```

## Evaluation

The tokenizer was evaluated on a held-out corpus of **3,000 samples from the Tashkeela corpus**. These samples are distinct from the data used to train the tokenizer.

The evaluation compares the original `google/gemma-3-1b-pt` tokenizer with the expanded `RanaGaber/gemma_1B_pt_diac` tokenizer.

| Metric           | `google/gemma-3-1b-pt` | `RanaGaber/gemma_1B_pt_diac` |
| ---------------- | ---------------------: | ---------------------------: |
| **Total Words**  |                244,114 |                      244,114 |
| **Total Tokens** |              1,019,110 |                  **343,313** |
| **Fertility**    |                  4.175 |                    **1.406** |

The expanded tokenizer reduces fertility from **4.175 to 1.406 tokens per word**, substantially reducing token fragmentation on the held-out corpus.

## Project Structure

```text
.
├── token_exp/
│   ├── new_vocab.py
│   ├── tokenizer_exp.py
│   ├── embedding_exp.py
│   ├── configs.py
│   ├── cpt.py
│   └── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

```bash
uv sync
```

## Usage

Update the arguments in `main.py`:

```python
if __name__ == "__main__":
    token = "your token"
    repo_id = "your id"
    model_id = "your model"
    txt_path = "your txt path"
    type = "LM"  # LM or Seq2Seq
    final_repo = "your repo id after cpt"
```

Then run:

```bash
uv run main.py
```

### Arguments

| Argument     | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| `txt_path`   | Path to the corpus used to train the new SentencePiece tokenizer and perform CPT |
| `model_id`   | Hugging Face model ID of the pretrained model                                    |
| `type`       | `"LM"` for causal language models or `"Seq2Seq"` for encoder-decoder models      |
| `token`      | Hugging Face access token used to upload the model                               |
| `repo_id`    | Hugging Face repository for the expanded model                                   |
| `final_repo` | Hugging Face repository for the final CPT model                                  |

## Continued Pretraining

After expanding the vocabulary and initializing the new embeddings, the tool automatically performs **Continued Pretraining (CPT)** using the same corpus.

This allows the newly added embeddings to be refined through exposure to their actual usage in the target corpus.

```text
Pretrained model
      ↓
Vocabulary expansion
      ↓
New embeddings initialized
      ↓
CPT on target corpus
      ↓
Final adapted model
```

For causal language models, CPT uses the standard causal language modeling objective.

## Example

```python
if __name__ == "__main__":
    token = "your token"
    repo_id = "your id"
    model_id = "your model"
    txt_path = "your txt path"

    type = "LM"  # LM or Seq2Seq

    final_repo = "your repo id after cpt"
```

The pipeline automatically:

1. Trains a SentencePiece tokenizer on `txt_path`.
2. Expands the pretrained tokenizer.
3. Resizes the model embeddings.
4. Initializes newly added embeddings from their old-token fragments.
5. Performs CPT on the same corpus.
6. Saves the resulting model and tokenizer.
7. Uploads the final model to Hugging Face.

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
```
