# Tokenizer Vocabulary Expansion

This project is designed to address the heavy token fragmentation often observed in low-resource languages.

For example, a pretrained tokenizer may tokenize a diacritized Arabic word into many small subtokens, effectively approaching character-level tokenization. We can train a new SentencePiece tokenizer on text from an under-resourced language, add its new vocabulary to the pretrained tokenizer, and initialize the embeddings of the newly added tokens using the pretrained model's existing embeddings.

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
Initialize new embeddings from old-token fragments
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
│   └── embedding_exp.py
├── main.py
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
    main(
        txt_path="path/to/training_corpus.txt",
        model_id="your/model",
        type="LM"
    )
```

Then run:

```bash
uv run main.py
```

### Arguments

| Argument   | Description                                                                 |
| ---------- | --------------------------------------------------------------------------- |
| `txt_path` | Path to the corpus used to train the new SentencePiece tokenizer            |
| `model_id` | Hugging Face model ID of the pretrained model                               |
| `type`     | `"LM"` for causal language models or `"Seq2Seq"` for encoder-decoder models |

## Example

```python
main(
    txt_path="data/arabic_corpus.txt",
    model_id="google/gemma-3-1b",
    type="LM"
)
```

## Saving the Expanded Model

The pipeline returns:

```python
model, tokenizer
```

The expanded tokenizer can be saved with:

```python
tokenizer.save_pretrained("expanded_tokenizer")
```

and the model with:

```python
model.save_pretrained("expanded_model")
```
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
