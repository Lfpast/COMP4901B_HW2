#!/usr/bin/env python3
"""Transformers-based inference script for IFEval benchmark.

This script uses HuggingFace Transformers to generate responses for IFEval prompts,
without requiring vLLM. It supports both CPU and GPU inference.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_input_data(input_path: str) -> List[Dict]:
    """Load the IFEval input data from jsonl file.

    Args:
        input_path: Path to input_data.jsonl

    Returns:
        List of dictionaries with prompt and metadata
    """
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
    """Apply HuggingFace chat template to a single prompt.

    Args:
        prompt: User prompt
        tokenizer: HuggingFace tokenizer with chat template
        system_message: Optional system message to prepend

    Returns:
        Formatted prompt ready for inference
    """
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
    """Generate a response for a single prompt.

    Args:
        prompt: Input prompt (already formatted with chat template)
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        top_p: Nucleus sampling parameter
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Generated response text
    """
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

    # Decode only the generated part (excluding input)
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
    batch_size: int = 1,
    system_message: str = None,
    device: str = None,
    dtype: str = "auto",
) -> None:
    """Run Transformers inference on IFEval dataset.

    Args:
        model_path: Path to the trained model
        input_data_path: Path to input_data.jsonl
        output_path: Path to save output responses in jsonl format
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        top_p: Nucleus sampling parameter
        batch_size: Currently not used (processes one at a time)
        system_message: Optional system message for chat template
        device: Device to use ('cuda', 'cpu', or None for auto)
        dtype: Model dtype ('auto', 'float16', 'bfloat16', 'float32')
    """
    logger.info("="*80)
    logger.info("Starting Transformers Inference for IFEval")
    logger.info("="*80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Input data: {input_data_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Max new tokens: {max_new_tokens}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Top-p: {top_p}")

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
    logger.info(f"Loading tokenizer from {model_path}")
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
    logger.info(f"Loading model from {model_path}")
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
    logger.info("Running inference...")
    results = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(tqdm(prompts, desc="Generating responses")):
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
                "prompt": prompt,  # Original prompt without chat template
                "response": response
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            results.append(result)

    logger.info("="*80)
    logger.info(f"Inference complete! Generated {len(results)} responses")
    logger.info(f"Results saved to: {output_path}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Transformers inference for IFEval benchmark (no vLLM required)"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to input_data.jsonl"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output responses (jsonl format)"
    )

    # Optional arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (currently not used, kept for compatibility)"
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default=None,
        help="Optional system message to prepend to prompts"
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

    run_inference(
        model_path=args.model_path,
        input_data_path=args.input_data,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        system_message=args.system_message,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
