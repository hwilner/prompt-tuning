# Model Zoo: Pretrained Soft Prompts

This page provides a collection of pretrained soft prompts for various tasks and datasets. You can use these as a starting point for your own fine-tuning or for direct inference.

## How to Use

1. **Download** the prompt file (`.pkl`).
2. **Load** the prompt in your script.
3. **Use** it with the corresponding model.

```python
from prompt_tuning.bert import train

# Load prompt
prompt_params = train.load_prompt("path/to/prompt.pkl")

# Use with model
# ...
```

## Available Prompts

### BERT-base-uncased

| Task | Dataset | Prompt Length | Accuracy | Download |
|---|---|---|---|---|
| Sentiment | SST-2 | 20 | 92.5% | [sst2_prompt.pkl](https://example.com/sst2_prompt.pkl) |
| NLI | MNLI | 30 | 84.1% | [mnli_prompt.pkl](https://example.com/mnli_prompt.pkl) |
| Paraphrase | QQP | 20 | 90.8% | [qqp_prompt.pkl](https://example.com/qqp_prompt.pkl) |

### Gemma-2B

| Task | Dataset | Prompt Length | Perplexity | Download |
|---|---|---|---|---|
| Generation | WikiText-2 | 50 | 25.1 | [wikitext2_prompt.pkl](https://example.com/wikitext2_prompt.pkl) |
| Summarization | CNN/DM | 100 | - | [cnndm_prompt.pkl](https://example.com/cnndm_prompt.pkl) |

### LLaMA-2-7B

| Task | Dataset | Prompt Length | Perplexity | Download |
|---|---|---|---|---|
| Generation | C4 | 50 | 19.8 | [c4_prompt.pkl](https://example.com/c4_prompt.pkl) |

## Training Your Own Prompts

See the tutorials for instructions on how to train your own prompts:

- [BERT Classification Tutorial](./tutorials/bert_classification.md)
- [Gemma Generation Tutorial](./tutorials/gemma_generation.md)

Once trained, you can save your prompt with:

```python
from prompt_tuning.bert import train

train.save_prompt(trained_prompt_params, "my_prompt.pkl")
```
