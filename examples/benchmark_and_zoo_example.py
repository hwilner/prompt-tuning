#!/usr/bin/env python3
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

"""Example usage of benchmarking suite and model zoo."""

from prompt_tuning import model_zoo


def example_model_zoo():
    """Example: Using the model zoo."""
    print("\n" + "="*60)
    print("Example 1: Model Zoo Usage")
    print("="*60 + "\n")
    
    # List all available prompts
    print("1. Listing all available prompts:")
    prompts = model_zoo.list_prompts()
    for prompt in prompts:
        print(f"  - {prompt['key']}: {prompt['description']}")
    
    print("\n2. Filtering prompts by model:")
    bert_prompts = model_zoo.list_prompts(model="bert-base-uncased")
    for prompt in bert_prompts:
        print(f"  - {prompt['key']}")
    
    print("\n3. Getting prompt information:")
    info = model_zoo.get_prompt_info("bert-base-uncased", "sst2")
    print(f"  Model: {info['model']}")
    print(f"  Task: {info['task']}")
    print(f"  Prompt Length: {info['prompt_length']}")
    print(f"  Accuracy: {info['accuracy']:.1%}")
    
    print("\n4. Loading a pretrained prompt:")
    try:
        prompt_data = model_zoo.load_prompt("bert-base-uncased", "sst2")
        print(f"  ✓ Loaded prompt for bert-base-uncased/sst2")
        print(f"  Metadata: {prompt_data.get('metadata', {})}")
    except Exception as e:
        print(f"  Note: {e}")
    
    print("\n5. Saving a custom prompt:")
    # Dummy prompt parameters
    dummy_prompt_params = {"prompt": {"prompt": None}}
    metadata = {
        "accuracy": 0.95,
        "prompt_length": 20,
        "description": "My custom prompt",
    }
    output_path = model_zoo.save_prompt(
        dummy_prompt_params,
        "bert-base-uncased",
        "custom_task",
        metadata=metadata,
    )
    print(f"  ✓ Saved to: {output_path}")


def example_benchmarking():
    """Example: Using the benchmarking suite."""
    print("\n" + "="*60)
    print("Example 2: Benchmarking Suite Usage")
    print("="*60 + "\n")
    
    print("To run a benchmark, use the command line:")
    print()
    print("  python -m prompt_tuning.benchmark \\")
    print("    --model_name_or_path=bert-base-uncased \\")
    print("    --task_name=sst2 \\")
    print("    --prompt_length=20 \\")
    print("    --method=prompt_tuning \\")
    print("    --num_epochs=3")
    print()
    print("This will:")
    print("  1. Load the BERT model and SST-2 dataset")
    print("  2. Create a prompt-tuned model")
    print("  3. Train for 3 epochs")
    print("  4. Evaluate on the test set")
    print("  5. Save results to benchmark_results/")
    print()
    print("Supported models:")
    print("  - bert-base-uncased, bert-large-uncased")
    print("  - gemma-2b, gemma-7b")
    print("  - llama-2-7b, llama-2-13b")
    print()
    print("Supported methods:")
    print("  - prompt_tuning (default)")
    print("  - p_tuning_v2 (deep prompts)")
    print("  - multitask_prompt_tuning")
    print("  - full_finetuning")


def example_integration():
    """Example: Integrating model zoo with your workflow."""
    print("\n" + "="*60)
    print("Example 3: Integration with Your Workflow")
    print("="*60 + "\n")
    
    print("Example code to use a pretrained prompt:")
    print()
    print("```python")
    print("from prompt_tuning import model_zoo")
    print("from prompt_tuning.bert import prompts")
    print("from transformers import FlaxBertForSequenceClassification")
    print()
    print("# Load pretrained prompt")
    print("prompt_data = model_zoo.load_prompt('bert-base-uncased', 'sst2')")
    print("prompt_params = prompt_data['prompt']")
    print()
    print("# Load base model")
    print("model = FlaxBertForSequenceClassification.from_pretrained('bert-base-uncased')")
    print()
    print("# Create prompt-tuned model")
    print("prompt_config = prompts.PromptConfig(prompt_length=20, embed_dim=768)")
    print("prompt_model = prompts.create_prompt_model(model, prompt_config)")
    print()
    print("# Use for inference")
    print("output = prompt_model.apply(")
    print("    {'params': {**model.params, **prompt_params}},")
    print("    input_ids,")
    print(")")
    print("```")


if __name__ == "__main__":
    example_model_zoo()
    example_benchmarking()
    example_integration()
    
    print("\n" + "="*60)
    print("For more information, see:")
    print("  - docs/model_zoo.md")
    print("  - docs/benchmarking.md")
    print("="*60 + "\n")
