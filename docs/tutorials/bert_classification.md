# Tutorial: Prompt Tuning for Text Classification with BERT

This tutorial demonstrates how to use prompt tuning to fine-tune a BERT model for text classification. We will use the GLUE SST-2 dataset (sentiment analysis).

## 1. Setup

First, install the necessary libraries:

```bash
pip install transformers[flax] datasets optax
```

## 2. Load Data

We use the `datasets` library to load the SST-2 dataset:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("glue", "sst2")

# Preprocess data
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

processed_dataset = dataset.map(preprocess_function, batched=True)
```

## 3. Create Prompt-Tuned Model

Next, we create our prompt-tuned BERT model:

```python
import jax
from transformers import FlaxBertForSequenceClassification
from prompt_tuning.bert import prompts

# Load base BERT model
model = FlaxBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Create prompt config
prompt_config = prompts.PromptConfig(
    prompt_length=20,
    embed_dim=768,  # BERT-base
)

# Create prompt-tuned model
prompt_model = prompts.PromptBERTForSequenceClassification(
    base_model=model,
    num_labels=2,
)
```

## 4. Initialize Parameters

We initialize the prompt parameters and freeze the base model:

```python
# Initialize prompt parameters
rng = jax.random.PRNGKey(42)
prompt_params = prompt_model.init_prompt_params(rng)

# Freeze base model parameters
base_params = model.params
```

## 5. Train

Now we can train the prompt:

```python
from prompt_tuning.bert import train

# Create data loader
# ... (code to create JAX-compatible data loader)

# Train
trained_prompt_params, history = train.train_prompt(
    model=prompt_model,
    prompt_params=prompt_params,
    base_params=base_params,
    train_ds=train_dataloader,
    eval_ds=eval_dataloader,
    num_epochs=3,
    learning_rate=0.3,
)
```

## 6. Evaluate

After training, we can evaluate the model:

```python
# Evaluate on test set
eval_metrics = train.evaluate(
    state=train_state,  # Create state from trained params
    eval_ds=test_dataloader,
)

print(f"Test Accuracy: {eval_metrics["accuracy"]:.4f}")
```

## 7. Save and Load Prompt

Finally, we can save the trained prompt for later use:

```python
# Save prompt
train.save_prompt(trained_prompt_params, "sst2_prompt.pkl")

# Load prompt
loaded_prompt_params = train.load_prompt("sst2_prompt.pkl")
```

## Full Example

See `examples/bert_classification.py` for a complete, runnable example.
