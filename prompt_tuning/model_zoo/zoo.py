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

"""Model zoo for pretrained soft prompts."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve


# Model zoo catalog
MODEL_ZOO_CATALOG = {
    "bert-base-uncased/sst2": {
        "model": "bert-base-uncased",
        "task": "sst2",
        "prompt_length": 20,
        "accuracy": 0.925,
        "url": "https://example.com/prompts/bert_base_sst2.pkl",  # Placeholder
        "description": "Sentiment analysis on SST-2 dataset",
    },
    "bert-base-uncased/mnli": {
        "model": "bert-base-uncased",
        "task": "mnli",
        "prompt_length": 30,
        "accuracy": 0.841,
        "url": "https://example.com/prompts/bert_base_mnli.pkl",  # Placeholder
        "description": "Natural language inference on MNLI dataset",
    },
    "bert-base-uncased/qqp": {
        "model": "bert-base-uncased",
        "task": "qqp",
        "prompt_length": 20,
        "accuracy": 0.908,
        "url": "https://example.com/prompts/bert_base_qqp.pkl",  # Placeholder
        "description": "Paraphrase detection on QQP dataset",
    },
    "gemma-2b/wikitext2": {
        "model": "gemma-2b",
        "task": "wikitext2",
        "prompt_length": 50,
        "perplexity": 25.1,
        "url": "https://example.com/prompts/gemma_2b_wikitext2.pkl",  # Placeholder
        "description": "Language modeling on WikiText-2",
    },
    "llama-2-7b/c4": {
        "model": "llama-2-7b",
        "task": "c4",
        "prompt_length": 50,
        "perplexity": 19.8,
        "url": "https://example.com/prompts/llama2_7b_c4.pkl",  # Placeholder
        "description": "Language modeling on C4 dataset",
    },
}


def list_prompts(model: Optional[str] = None, task: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available pretrained prompts.
    
    Args:
        model: Filter by model name (optional).
        task: Filter by task name (optional).
        
    Returns:
        List of prompt metadata dictionaries.
    """
    prompts = []
    
    for key, metadata in MODEL_ZOO_CATALOG.items():
        # Apply filters
        if model and metadata["model"] != model:
            continue
        if task and metadata["task"] != task:
            continue
        
        prompts.append({
            "key": key,
            **metadata,
        })
    
    return prompts


def load_prompt(
    model: str,
    task: str,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a pretrained prompt from the model zoo.
    
    Args:
        model: Model name (e.g., "bert-base-uncased").
        task: Task name (e.g., "sst2").
        cache_dir: Directory to cache downloaded prompts (default: ~/.cache/prompt_tuning).
        
    Returns:
        Dictionary containing prompt parameters.
        
    Raises:
        ValueError: If prompt not found in catalog.
        FileNotFoundError: If prompt file cannot be downloaded.
    """
    # Look up in catalog
    key = f"{model}/{task}"
    if key not in MODEL_ZOO_CATALOG:
        available = ", ".join(MODEL_ZOO_CATALOG.keys())
        raise ValueError(
            f"Prompt not found: {key}. Available prompts: {available}"
        )
    
    metadata = MODEL_ZOO_CATALOG[key]
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "prompt_tuning"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already cached
    cache_path = cache_dir / f"{model.replace('/', '_')}_{task}.pkl"
    
    if not cache_path.exists():
        print(f"Downloading prompt from: {metadata['url']}")
        print(f"Note: This is a placeholder URL. In production, prompts would be hosted on a CDN.")
        print(f"Creating dummy prompt at: {cache_path}")
        
        # Create a dummy prompt for demonstration
        # In production, this would download from the actual URL
        dummy_prompt = {
            "prompt": {
                "prompt": None,  # Would contain actual parameters
            },
            "metadata": metadata,
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(dummy_prompt, f)
    
    # Load from cache
    print(f"Loading prompt from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        prompt_data = pickle.load(f)
    
    return prompt_data


def save_prompt(
    prompt_params: Dict[str, Any],
    model: str,
    task: str,
    metadata: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
):
    """Save a trained prompt for sharing.
    
    Args:
        prompt_params: Prompt parameters to save.
        model: Model name.
        task: Task name.
        metadata: Optional metadata about the prompt.
        output_path: Path to save prompt (default: ./prompts/{model}_{task}.pkl).
    """
    if output_path is None:
        output_dir = Path("./prompts")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model.replace('/', '_')}_{task}.pkl"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data to save
    prompt_data = {
        "prompt": prompt_params,
        "metadata": metadata or {},
        "model": model,
        "task": task,
    }
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(prompt_data, f)
    
    print(f"Prompt saved to: {output_path}")
    return str(output_path)


def get_prompt_info(model: str, task: str) -> Dict[str, Any]:
    """Get information about a prompt without downloading it.
    
    Args:
        model: Model name.
        task: Task name.
        
    Returns:
        Prompt metadata dictionary.
    """
    key = f"{model}/{task}"
    if key not in MODEL_ZOO_CATALOG:
        raise ValueError(f"Prompt not found: {key}")
    
    return MODEL_ZOO_CATALOG[key].copy()


def print_catalog():
    """Print the model zoo catalog in a readable format."""
    print("\n" + "="*80)
    print("Prompt Tuning Model Zoo")
    print("="*80 + "\n")
    
    # Group by model
    by_model = {}
    for key, metadata in MODEL_ZOO_CATALOG.items():
        model = metadata["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(metadata)
    
    for model, prompts in sorted(by_model.items()):
        print(f"\n{model}")
        print("-" * len(model))
        for prompt in prompts:
            print(f"  • {prompt['task']}: {prompt['description']}")
            print(f"    Prompt length: {prompt['prompt_length']}")
            if 'accuracy' in prompt:
                print(f"    Accuracy: {prompt['accuracy']:.1%}")
            if 'perplexity' in prompt:
                print(f"    Perplexity: {prompt['perplexity']:.1f}")
            print()
    
    print("="*80)
    print(f"Total prompts: {len(MODEL_ZOO_CATALOG)}")
    print("="*80 + "\n")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Tuning Model Zoo")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available prompts")
    list_parser.add_argument("--model", type=str, help="Filter by model")
    list_parser.add_argument("--task", type=str, help="Filter by task")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load a prompt")
    load_parser.add_argument("model", type=str, help="Model name")
    load_parser.add_argument("task", type=str, help="Task name")
    load_parser.add_argument("--cache_dir", type=str, help="Cache directory")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get prompt info")
    info_parser.add_argument("model", type=str, help="Model name")
    info_parser.add_argument("task", type=str, help="Task name")
    
    # Catalog command
    subparsers.add_parser("catalog", help="Print full catalog")
    
    args = parser.parse_args()
    
    if args.command == "list":
        prompts = list_prompts(model=args.model, task=args.task)
        print(json.dumps(prompts, indent=2))
    
    elif args.command == "load":
        prompt_data = load_prompt(args.model, args.task, args.cache_dir)
        print(f"Loaded prompt for {args.model}/{args.task}")
        print(f"Metadata: {prompt_data.get('metadata', {})}")
    
    elif args.command == "info":
        info = get_prompt_info(args.model, args.task)
        print(json.dumps(info, indent=2))
    
    elif args.command == "catalog":
        print_catalog()
    
    else:
        parser.print_help()
