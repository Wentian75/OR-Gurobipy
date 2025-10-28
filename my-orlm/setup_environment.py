#!/usr/bin/env python3
"""
Environment Setup Script for ORLM QLoRA Fine-tuning
This script checks and installs all required dependencies
"""

import subprocess
import sys
import os


def run_command(cmd, description, check=True):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f">>> {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=check,
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}", file=sys.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        if not check:
            return False
        sys.exit(1)


def check_gpu():
    """Check if NVIDIA GPU is available"""
    print("\n" + "="*60)
    print("Checking GPU availability...")
    print("="*60)

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ PyTorch version: {torch.__version__}")
            return True
        else:
            print("⚠ Warning: No GPU detected. Training will be very slow on CPU.")
            return False
    except ImportError:
        print("PyTorch not installed yet, will check GPU after installation.")
        return None


def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("\n" + "="*60)
    print("Installing PyTorch with CUDA 12.1 support...")
    print("="*60)

    cmd = (
        "pip install --upgrade pip && "
        "pip install --index-url https://download.pytorch.org/whl/cu121 "
        "torch torchvision torchaudio"
    )
    return run_command(cmd, "Installing PyTorch with CUDA support")


def install_qlora_dependencies():
    """Install QLoRA training dependencies"""
    print("\n" + "="*60)
    print("Installing QLoRA dependencies...")
    print("="*60)

    packages = [
        '"transformers>=4.43"',
        '"peft>=0.11.0"',
        '"bitsandbytes>=0.43.1"',
        '"accelerate>=0.31.0"',
        '"datasets>=2.19"',
        '"trl>=0.9.4"',
        'sentencepiece',
        '"protobuf<5"',
    ]

    cmd = f"pip install {' '.join(packages)}"
    return run_command(cmd, "Installing QLoRA dependencies")


def disable_wandb():
    """Disable Weights & Biases"""
    print("\n" + "="*60)
    print("Disabling W&B (Weights & Biases)...")
    print("="*60)

    # Uninstall wandb if present
    run_command("pip uninstall -y wandb", "Uninstalling W&B", check=False)

    # Set environment variables
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print("✓ W&B disabled")
    print("✓ Environment variables set")


def verify_installation():
    """Verify all installations"""
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)

    try:
        import torch
        import transformers
        import peft
        import bitsandbytes
        import accelerate
        import datasets
        import trl

        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Transformers: {transformers.__version__}")
        print(f"✓ PEFT: {peft.__version__}")
        print(f"✓ Bitsandbytes: {bitsandbytes.__version__}")
        print(f"✓ Accelerate: {accelerate.__version__}")
        print(f"✓ Datasets: {datasets.__version__}")
        print(f"✓ TRL: {trl.__version__}")

        if torch.cuda.is_available():
            print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("\n⚠ Warning: CUDA not available")

        print("\n" + "="*60)
        print("✓ All dependencies installed successfully!")
        print("="*60)
        return True

    except ImportError as e:
        print(f"\n✗ Installation verification failed: {e}")
        return False


def main():
    """Main setup function"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   ORLM QLoRA Fine-tuning Environment Setup               ║
    ║   This script will install all required dependencies     ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Check initial GPU status
    check_gpu()

    # Install PyTorch
    if not install_pytorch():
        print("\n✗ Failed to install PyTorch")
        sys.exit(1)

    # Install QLoRA dependencies
    if not install_qlora_dependencies():
        print("\n✗ Failed to install QLoRA dependencies")
        sys.exit(1)

    # Disable W&B
    disable_wandb()

    # Verify installation
    if not verify_installation():
        print("\n✗ Installation verification failed")
        sys.exit(1)

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   ✓ Setup Complete!                                      ║
    ║   You can now run the training pipeline                  ║
    ╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
