# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple text classification example with Gemma prompt tuning.

This example demonstrates how to use prompt tuning with Gemma for a simple
text classification task. It shows:

1. Loading a pretrained Gemma model
2. Creating a prompt-tuned version
3. Training the prompt on a classification dataset
4. Evaluating the results

Example usage:
  python simple_classification.py \\
    --model_size=2b \\
    --prompt_length=20 \\
    --num_epochs=3 \\
    --learning_rate=0.3
"""

import argparse
from typing import Dict, Iterator

import jax
import jax.numpy as jnp
from gemma import gm

from prompt_tuning.gemma import prompts, train


def create_dummy_dataset(
    num_samples: int = 1000,
    seq_len: int = 128,
    vocab_size: int = 256000,
    batch_size: int = 8,
) -> Iterator[Dict[str, jnp.ndarray]]:
  """Create a dummy dataset for demonstration.
  
  In practice, replace this with your actual dataset loading logic.
  
  Args:
    num_samples: Number of samples in the dataset.
    seq_len: Sequence length.
    vocab_size: Vocabulary size.
    batch_size: Batch size.
    
  Yields:
    Batches of data with 'input_ids' and 'labels'.
  """
  rng = jax.random.PRNGKey(0)
  
  for i in range(0, num_samples, batch_size):
    rng, input_rng, label_rng = jax.random.split(rng, 3)
    
    # Generate random input IDs
    input_ids = jax.random.randint(
        input_rng,
        (batch_size, seq_len),
        0,
        vocab_size,
    )
    
    # Generate random labels (shifted input for language modeling)
    labels = jnp.roll(input_ids, -1, axis=1)
    
    yield {
        'input_ids': input_ids,
        'labels': labels,
    }


def main(args):
  """Main training function."""
  
  print("=" * 80)
  print("Gemma Prompt Tuning - Simple Classification Example")
  print("=" * 80)
  
  # Step 1: Load pretrained Gemma model
  print("\n1. Loading Gemma model...")
  
  if args.model_size == '2b':
    model = gm.nn.Gemma3_2B()
    embed_dim = 2048
    checkpoint_path = gm.ckpts.CheckpointPath.GEMMA3_2B_IT
  elif args.model_size == '4b':
    model = gm.nn.Gemma3_4B()
    embed_dim = 2816
    checkpoint_path = gm.ckpts.CheckpointPath.GEMMA3_4B_IT
  else:
    raise ValueError(f"Unsupported model size: {args.model_size}")
  
  print(f"Loading checkpoint: {checkpoint_path}")
  base_params = gm.ckpts.load_params(checkpoint_path)
  print("✓ Model loaded successfully")
  
  # Step 2: Create prompt-tuned model
  print(f"\n2. Creating prompt-tuned model (prompt_length={args.prompt_length})...")
  
  prompt_config = prompts.PromptConfig(
      prompt_length=args.prompt_length,
      embed_dim=embed_dim,
      init_scale=0.5,
  )
  
  prompt_model = prompts.create_prompt_model(model, prompt_config)
  print("✓ Prompt model created")
  
  # Step 3: Initialize prompt parameters
  print("\n3. Initializing prompt parameters...")
  
  rng = jax.random.PRNGKey(args.seed)
  prompt_params = prompt_model.init_prompt_params(rng)
  
  num_params = sum(p.size for p in jax.tree_util.tree_leaves(prompt_params))
  print(f"✓ Prompt parameters initialized ({num_params:,} parameters)")
  
  # Step 4: Create datasets
  print("\n4. Creating datasets...")
  
  train_ds = create_dummy_dataset(
      num_samples=args.train_samples,
      batch_size=args.batch_size,
  )
  
  eval_ds = create_dummy_dataset(
      num_samples=args.eval_samples,
      batch_size=args.batch_size,
  )
  
  print(f"✓ Train samples: {args.train_samples}")
  print(f"✓ Eval samples: {args.eval_samples}")
  print(f"✓ Batch size: {args.batch_size}")
  
  # Step 5: Train prompt
  print(f"\n5. Training prompt for {args.num_epochs} epochs...")
  print("-" * 80)
  
  trained_prompt_params, history = train.train_prompt(
      model=prompt_model,
      prompt_params=prompt_params,
      base_params=base_params,
      train_ds=train_ds,
      eval_ds=eval_ds,
      num_epochs=args.num_epochs,
      steps_per_epoch=args.steps_per_epoch,
      learning_rate=args.learning_rate,
      weight_decay=args.weight_decay,
  )
  
  print("-" * 80)
  print("✓ Training completed")
  
  # Step 6: Save prompt
  if args.output_path:
    print(f"\n6. Saving prompt to {args.output_path}...")
    train.save_prompt(trained_prompt_params, args.output_path)
    print("✓ Prompt saved")
  
  # Print final results
  print("\n" + "=" * 80)
  print("Training Summary")
  print("=" * 80)
  print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
  print(f"Final Train Accuracy: {history['train_accuracy'][-1]:.4f}")
  if history['eval_loss']:
    print(f"Final Eval Loss: {history['eval_loss'][-1]:.4f}")
    print(f"Final Eval Accuracy: {history['eval_accuracy'][-1]:.4f}")
  print("=" * 80)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Train Gemma with prompt tuning for text classification'
  )
  
  # Model arguments
  parser.add_argument(
      '--model_size',
      type=str,
      default='2b',
      choices=['2b', '4b'],
      help='Gemma model size',
  )
  parser.add_argument(
      '--prompt_length',
      type=int,
      default=20,
      help='Number of prompt tokens',
  )
  
  # Training arguments
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=3,
      help='Number of training epochs',
  )
  parser.add_argument(
      '--steps_per_epoch',
      type=int,
      default=None,
      help='Steps per epoch (None = full dataset)',
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.3,
      help='Learning rate',
  )
  parser.add_argument(
      '--weight_decay',
      type=float,
      default=0.0,
      help='Weight decay',
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=8,
      help='Batch size',
  )
  
  # Data arguments
  parser.add_argument(
      '--train_samples',
      type=int,
      default=1000,
      help='Number of training samples',
  )
  parser.add_argument(
      '--eval_samples',
      type=int,
      default=200,
      help='Number of evaluation samples',
  )
  
  # Other arguments
  parser.add_argument(
      '--seed',
      type=int,
      default=42,
      help='Random seed',
  )
  parser.add_argument(
      '--output_path',
      type=str,
      default='gemma_prompt.pkl',
      help='Path to save trained prompt',
  )
  
  args = parser.parse_args()
  main(args)
