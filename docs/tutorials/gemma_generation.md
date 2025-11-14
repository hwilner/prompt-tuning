# Tutorial: Prompt Tuning for Text Generation with Gemma

This tutorial demonstrates how to use prompt tuning to fine-tune a Gemma model for text generation.

## 1. Setup

First, install the necessary libraries:

```bash
pip install gemma optax
```

## 2. Load Model

We load the pretrained Gemma model:

```python
from gemma import gm

# Load Gemma 2B model
model = gm.nn.Gemma3_2B()
base_params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_2B_IT)
```

## 3. Create Prompt-Tuned Model

Next, we create our prompt-tuned Gemma model:

```python
import jax
from prompt_tuning.gemma import prompts

# Create prompt config
prompt_config = prompts.PromptConfig(
    prompt_length=50,  # Longer for generation
    embed_dim=2048,  # Gemma 2B
)

# Create prompt-tuned model
prompt_model = prompts.create_prompt_model(model, prompt_config)
```

## 4. Initialize Parameters

We initialize the prompt parameters:

```python
# Initialize prompt parameters
rng = jax.random.PRNGKey(42)
prompt_params = prompt_model.init_prompt_params(rng)
```

## 5. Train

Now we can train the prompt on a generation task:

```python
from prompt_tuning.gemma import train

# Create data loader for generation
# ... (code to create JAX-compatible data loader)

# Train
trained_prompt_params, history = train.train_prompt(
    model=prompt_model,
    prompt_params=prompt_params,
    base_params=base_params,
    train_ds=train_dataloader,
    num_epochs=3,
    learning_rate=0.1,
)
```

## 6. Generate Text

After training, we can use the prompt to generate text:

```python
# Combine params
full_params = {**base_params, **trained_prompt_params}

# Input text
input_text = "The meaning of life is"

# Tokenize
# ... (code to tokenize input_text)

# Generate
output_ids = prompt_model.apply(
    {"params": full_params},
    input_ids,
    method=prompt_model.generate,  # Use generate method
    max_length=100,
)

# Decode
# ... (code to decode output_ids)
```

## Full Example

See `examples/gemma_generation.py` for a complete, runnable example.
