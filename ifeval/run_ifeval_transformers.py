#!/usr/bin/env python3
"""Unified pipeline for IFEval using Transformers (no vLLM required).

This script provides a complete pipeline to:
1. Run inference using Transformers on a trained model
2. Evaluate the generated responses using IFEval metrics
3. Report results

Usage:
    # Run both inference and evaluation
    python run_ifeval_transformers.py --mode all --model_path /path/to/model

    # Run inference only
    python run_ifeval_transformers.py --mode inference --model_path /path/to/model

    # Run evaluation only (requires existing response file)
    python run_ifeval_transformers.py --mode eval --input_response_data /path/to/responses.jsonl
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add paths for evaluation imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both parent directory and current directory to path
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Create an alias so instruction_following_eval imports work
folder_name = os.path.basename(current_dir)
sys.modules['instruction_following_eval'] = __import__(folder_name)

import evaluation_lib


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_non_reasoning_content(
    text: str,
    think_start_token: str = '<think>',
    think_end_token: str = '</think>',
) -> str:
    """Strip reasoning segments wrapped in think tags from model output."""
    # When only the end token is present, keep the trailing content.
    if think_start_token not in text and think_end_token in text:
        return text.split(think_end_token)[-1].strip()

    reasoning_regex = re.compile(rf'{think_start_token}(.*?){think_end_token}',
                                 re.DOTALL)
    cleaned_text = reasoning_regex.sub('', text)

    # If there is an unmatched start token, drop everything after it.
    if think_start_token in cleaned_text and think_end_token not in cleaned_text:
        cleaned_text = cleaned_text.split(think_start_token)[0]

    return cleaned_text.strip()


def load_input_data(input_path: str) -> List[Dict]:
    """Load the IFEval input data from jsonl file."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(data)} prompts from {input_path}")
    return data


def apply_chat_template(
    prompt: str,
    tokenizer: AutoTokenizer,
    system_message: str = None
) -> str:
    """Apply HuggingFace chat template to a single prompt."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    # Apply chat template
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
        return prompt


def generate_response(
    prompt: str,
    model,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "cuda"
) -> str:
    """Generate a response for a single prompt."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate
    with torch.no_grad():
        if temperature == 0.0:
            # Greedy decoding
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            # Sampling
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # Decode the entire output (vLLM only returns the generated part)
    # So we need to extract only the newly generated tokens
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def run_inference(
    model_path: str,
    input_data_path: str,
    output_path: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    system_message: str = None,
    device: str = None,
    dtype: str = "auto",
) -> str:
    """Run Transformers inference and return path to response file."""
    logger.info("="*80)
    logger.info("STEP 1: Running Transformers Inference")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Input: {input_data_path}")
    logger.info(f"Output: {output_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Determine dtype
    torch_dtype = torch.float32
    if dtype == "auto":
        if device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32
    logger.info(f"Using dtype: {torch_dtype}")

    # Load input data
    input_data = load_input_data(input_data_path)
    prompts = [item["prompt"] for item in input_data]

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    logger.info("Model loaded successfully")

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    logger.info("Generating responses...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            # Apply chat template
            formatted_prompt = apply_chat_template(prompt, tokenizer, system_message)
            
            # Generate response
            try:
                response = generate_response(
                    formatted_prompt,
                    model,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    device=device
                )
            except Exception as e:
                logger.error(f"Error generating response for prompt {i}: {e}")
                response = ""
            
            # Save result
            result = {
                "prompt": prompt,
                "response": response
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info(f"✓ Inference complete! {len(prompts)} responses saved to {output_path}")
    return output_path


def run_evaluation(
    input_data_path: str,
    input_response_path: str,
    output_dir: str
) -> Dict[str, float]:
    """Run IFEval evaluation and return accuracy scores."""
    logger.info("="*80)
    logger.info("STEP 2: Running IFEval Evaluation")
    logger.info("="*80)
    logger.info(f"Input data: {input_data_path}")
    logger.info(f"Responses: {input_response_path}")
    logger.info(f"Output dir: {output_dir}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    inputs = evaluation_lib.read_prompt_list(input_data_path)
    prompt_to_response = evaluation_lib.read_prompt_to_response_dict(input_response_path)
    cleaned_prompt_to_response = {
        prompt: extract_non_reasoning_content(response)
        for prompt, response in prompt_to_response.items()
    }

    results = {}

    # Run both strict and loose evaluation
    for func, mode in [
        (evaluation_lib.test_instruction_following_strict, "strict"),
        (evaluation_lib.test_instruction_following_loose, "loose"),
    ]:
        logger.info(f"Evaluating with {mode} mode...")
        outputs = []
        for inp in inputs:
            outputs.append(func(inp, cleaned_prompt_to_response))

        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        results[mode] = accuracy

        # Save results
        output_file = os.path.join(output_dir, f"eval_results_{mode}.jsonl")
        evaluation_lib.write_outputs(output_file, outputs)

        logger.info(f"✓ {mode.capitalize()} evaluation complete!")
        logger.info(f"  Accuracy: {accuracy:.4f} ({sum(follow_all_instructions)}/{len(outputs)})")
        logger.info(f"  Results saved to: {output_file}")

    return results


def save_summary_json(results: Dict[str, float], output_path: str, response_file: str):
    """Save final scores to a JSON file."""
    summary = {
        "response_file": response_file,
        "strict_accuracy": results['strict'],
        "loose_accuracy": results['loose'],
        "strict_accuracy_percentage": results['strict'] * 100,
        "loose_accuracy_percentage": results['loose'] * 100
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Summary saved to: {output_path}")


def print_summary(results: Dict[str, float], response_file: str):
    """Print final summary of results."""
    logger.info("="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Response file: {response_file}")
    logger.info("")
    logger.info(f"Strict Accuracy:  {results['strict']:.4f} ({results['strict']*100:.2f}%)")
    logger.info(f"Loose Accuracy:   {results['loose']:.4f} ({results['loose']*100:.2f}%)")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="IFEval Pipeline using Transformers (no vLLM required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (inference + eval)
  python run_ifeval_transformers.py --mode all --model_path /path/to/model

  # Run inference only
  python run_ifeval_transformers.py --mode inference --model_path /path/to/model --output_dir ./results

  # Run evaluation only
  python run_ifeval_transformers.py --mode eval --input_response_data ./results/responses.jsonl
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "inference", "eval"],
        default="all",
        help="Pipeline mode: 'all' (inference+eval), 'inference' only, or 'eval' only"
    )

    # Model and data paths
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model (required for inference)"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="./data/sampled_input_data.jsonl",
        help="Path to IFEval input_data.jsonl"
    )
    parser.add_argument(
        "--input_response_data",
        type=str,
        help="Path to existing response data (for eval-only mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save all outputs"
    )

    # Inference parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default=None,
        help="Optional system message for chat template"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", None],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode in ["all", "inference"] and not args.model_path:
        parser.error("--model_path is required for inference mode")

    if args.mode == "eval" and not args.input_response_data:
        parser.error("--input_response_data is required for eval-only mode")

    # Ensure input_data exists
    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f"Input data not found: {args.input_data}")

    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    response_file = args.input_response_data
    if args.mode in ["all", "inference"]:
        # Generate response filename based on model name
        model_name = Path(args.model_path).name
        response_file = str(output_dir / f"responses_{model_name}.jsonl")

    # Run pipeline
    try:
        if args.mode in ["all", "inference"]:
            response_file = run_inference(
                model_path=args.model_path,
                input_data_path=args.input_data,
                output_path=response_file,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                system_message=args.system_message,
                device=args.device,
                dtype=args.dtype,
            )

        if args.mode in ["all", "eval"]:
            results = run_evaluation(
                input_data_path=args.input_data,
                input_response_path=response_file,
                output_dir=str(output_dir)
            )

            # Save summary JSON
            summary_path = str(output_dir / "summary.json")
            save_summary_json(results, summary_path, response_file)

            if args.mode == "all":
                print_summary(results, response_file)

        logger.info("✓ Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
