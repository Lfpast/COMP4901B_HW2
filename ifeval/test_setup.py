#!/usr/bin/env python3
"""Test script to verify the Transformers-based IFEval setup."""

import sys
import os

def check_imports():
    """Check if all required packages are installed."""
    print("Checking required packages...")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is NOT installed")
            missing.append(module)
    
    return missing


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("✗ CUDA is NOT available (will use CPU)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")


def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")
    
    required_files = [
        'run_ifeval_transformers.py',
        'inference_transformers.py',
        'run_transformers.sh',
        'evaluation_lib.py',
        'data/sampled_input_data.jsonl'
    ]
    
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ {filepath} exists")
        else:
            print(f"✗ {filepath} NOT found")


def main():
    print("="*60)
    print("IFEval Transformers Setup Test")
    print("="*60)
    
    missing = check_imports()
    check_cuda()
    check_files()
    
    print("\n" + "="*60)
    if missing:
        print("Setup incomplete!")
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    else:
        print("✓ Setup complete! Ready to run IFEval.")
        print("\nQuick start:")
        print("  bash run_transformers.sh ../SmolLM2-135M results/base")
    print("="*60)


if __name__ == "__main__":
    main()
