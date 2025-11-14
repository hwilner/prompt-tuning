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

"""Main script for running benchmarks."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from prompt_tuning.benchmark import runner


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt tuning benchmarks"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name or path (e.g., bert-base-uncased, gemma-2b)",
    )
    
    # Task arguments
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name (e.g., sst2, mnli, wikitext2)",
    )
    
    # Prompt tuning arguments
    parser.add_argument(
        "--method",
        type=str,
        default="prompt_tuning",
        choices=["prompt_tuning", "p_tuning_v2", "multitask_prompt_tuning", "full_finetuning"],
        help="Fine-tuning method to use",
    )
    
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=20,
        help="Length of soft prompts",
    )
    
    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.3,
        help="Learning rate for prompt tuning",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides num_epochs)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save results",
    )
    
    parser.add_argument(
        "--save_prompts",
        action="store_true",
        help="Save trained prompts",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    print(f"\n{'='*60}")
    print(f"Running benchmark: {args.model_name_or_path} on {args.task_name}")
    print(f"Method: {args.method}, Prompt Length: {args.prompt_length}")
    print(f"{'='*60}\n")
    
    try:
        results = runner.run_benchmark(
            model_name_or_path=args.model_name_or_path,
            task_name=args.task_name,
            method=args.method,
            prompt_length=args.prompt_length,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            seed=args.seed,
            save_prompts=args.save_prompts,
            output_dir=str(output_dir),
        )
        
        # Print results
        print(f"\n{'='*60}")
        print("Benchmark Results:")
        print(f"{'='*60}")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print(f"{'='*60}\n")
        
        # Save results to JSON
        results_file = output_dir / f"results_{args.model_name_or_path.replace('/', '_')}_{args.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\nError running benchmark: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
