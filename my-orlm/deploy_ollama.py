#!/usr/bin/env python3
"""
Deploy Model to Ollama
Creates Ollama Modelfile and deploys the GGUF model for local inference
"""

import os
import sys
import subprocess
from pathlib import Path


class OllamaDeployer:
    """Deploy GGUF model to Ollama"""

    def __init__(self, gguf_path, model_name='orlm-qwen3-8b'):
        """
        Initialize deployer

        Args:
            gguf_path: Path to GGUF model file
            model_name: Name for the Ollama model
        """
        self.gguf_path = Path(gguf_path)
        self.model_name = model_name
        self.modelfile_path = self.gguf_path.parent / 'Modelfile'

    def check_ollama(self):
        """Check if Ollama is installed"""
        print("\n" + "="*60)
        print("Checking Ollama installation...")
        print("="*60)

        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Ollama is installed")
            print(f"  Version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Ollama is not installed")
            print("\nTo install Ollama:")
            print("  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh")
            print("  Or visit: https://ollama.com/download")
            return False

    def check_gguf(self):
        """Check if GGUF file exists"""
        print("\n" + "="*60)
        print("Checking GGUF file...")
        print("="*60)

        if not self.gguf_path.exists():
            print(f"✗ Error: GGUF file not found: {self.gguf_path}")
            print("  Please run GGUF conversion first")
            sys.exit(1)

        size_gb = self.gguf_path.stat().st_size / (1024**3)
        print(f"✓ GGUF file found: {self.gguf_path}")
        print(f"  Size: {size_gb:.2f} GB")

    def create_modelfile(self):
        """Create Ollama Modelfile"""
        print("\n" + "="*60)
        print("Creating Modelfile...")
        print("="*60)

        # Template for OR problems
        template = '''Below is an operations research question. Build a mathematical model and corresponding Python code using `gurobipy` that appropriately addresses the question.

# Question:
{{ .Prompt }}

# Response:
'''

        # Modelfile content
        modelfile_content = f'''FROM {self.gguf_path.absolute()}

TEMPLATE """{template}"""

PARAMETER stop "</s>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "This script provides an example"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.2
PARAMETER repeat_last_n 256
PARAMETER num_ctx 8192
PARAMETER num_predict 2048

SYSTEM """You are an expert in operations research and optimization. You help users formulate and solve optimization problems using mathematical models and Python code with gurobipy."""
'''

        # Write Modelfile
        with open(self.modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)

        print(f"✓ Modelfile created: {self.modelfile_path}")
        print("\nModelfile content:")
        print("-" * 60)
        print(modelfile_content)
        print("-" * 60)

    def create_ollama_model(self):
        """Create Ollama model from Modelfile"""
        print("\n" + "="*60)
        print(f"Creating Ollama model: {self.model_name}")
        print("="*60)
        print("(This may take a few minutes...)")

        try:
            result = subprocess.run(
                ['ollama', 'create', '--force', self.model_name,
                 '-f', str(self.modelfile_path)],
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ Ollama model created successfully")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create Ollama model: {e}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            sys.exit(1)

    def list_models(self):
        """List available Ollama models"""
        print("\n" + "="*60)
        print("Available Ollama models:")
        print("="*60)

        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to list models: {e}")

    def test_model(self):
        """Test the deployed model"""
        print("\n" + "="*60)
        print("Testing model...")
        print("="*60)

        test_prompt = "A factory can produce two products A and B. Product A yields $50 profit and requires 2 hours. Product B yields $60 profit and requires 3 hours. There are 100 hours available. Formulate an optimization problem."

        print(f"\nTest prompt:\n{test_prompt}\n")
        print("Generating response (this may take a moment)...")
        print("-" * 60)

        try:
            # Run ollama
            process = subprocess.Popen(
                ['ollama', 'run', self.model_name, test_prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Stream output
            for line in process.stdout:
                print(line, end='')

            process.wait()

            if process.returncode == 0:
                print("\n" + "-" * 60)
                print("✓ Model test successful!")
            else:
                print("\n" + "-" * 60)
                print("⚠ Test completed with warnings")

        except Exception as e:
            print(f"\n✗ Test failed: {e}")

    def deploy(self, test=True):
        """
        Execute full deployment pipeline

        Args:
            test: Whether to test the model after deployment
        """
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║   Deploying Model to Ollama                              ║
        ╚══════════════════════════════════════════════════════════╝
        """)

        # Check Ollama
        if not self.check_ollama():
            sys.exit(1)

        # Check GGUF
        self.check_gguf()

        # Create Modelfile
        self.create_modelfile()

        # Create Ollama model
        self.create_ollama_model()

        # List models
        self.list_models()

        # Test model
        if test:
            self.test_model()

        print(f"""
        ╔══════════════════════════════════════════════════════════╗
        ║   ✓ Deployment Complete!                                 ║
        ║   Model name: {self.model_name:43s} ║
        ║                                                          ║
        ║   To use the model:                                      ║
        ║   $ ollama run {self.model_name:43s} ║
        ║                                                          ║
        ║   To remove the model:                                   ║
        ║   $ ollama rm {self.model_name:44s} ║
        ╚══════════════════════════════════════════════════════════╝
        """)


def find_gguf_file(gguf_dir='ORLM/checkpoints/gguf'):
    """
    Automatically find GGUF file in directory
    Prefers quantized models (Q5_K_M, Q4_K_M, etc.) over f16

    Args:
        gguf_dir: Directory to search for GGUF files

    Returns:
        Path to GGUF file, or None if not found
    """
    gguf_path = Path(gguf_dir)

    if not gguf_path.exists():
        return None

    # Priority order: quantized models first, then f16
    priority_patterns = [
        'model-Q5_K_M.gguf',
        'model-Q4_K_M.gguf',
        'model-Q8_0.gguf',
        'model-Q5_0.gguf',
        'model-Q4_0.gguf',
        'model-f16.gguf',
    ]

    # Try priority patterns first
    for pattern in priority_patterns:
        candidate = gguf_path / pattern
        if candidate.exists():
            return str(candidate)

    # If no priority match, find any .gguf file
    gguf_files = list(gguf_path.glob('*.gguf'))
    if gguf_files:
        # Sort by modification time, newest first
        gguf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(gguf_files[0])

    return None


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Deploy GGUF model to Ollama'
    )
    parser.add_argument('--gguf_path', type=str,
                       default=None,
                       help='Path to GGUF model file (auto-detected if not provided)')
    parser.add_argument('--gguf_dir', type=str,
                       default='ORLM/checkpoints/gguf',
                       help='Directory to search for GGUF files (default: ORLM/checkpoints/gguf)')
    parser.add_argument('--model_name', type=str,
                       default='orlm-qwen3-8b',
                       help='Name for the Ollama model')
    parser.add_argument('--no-test', dest='test', action='store_false',
                       help='Skip testing the model')
    parser.add_argument('--test', dest='test', action='store_true',
                       default=True,
                       help='Test the model after deployment (default)')

    args = parser.parse_args()

    # Auto-detect GGUF file if not provided
    gguf_path = args.gguf_path
    if not gguf_path:
        print("No GGUF path specified, auto-detecting...")
        gguf_path = find_gguf_file(args.gguf_dir)
        if not gguf_path:
            print(f"✗ Error: No GGUF file found in {args.gguf_dir}")
            print("  Please run GGUF conversion first or specify --gguf_path")
            sys.exit(1)
        print(f"✓ Auto-detected GGUF file: {gguf_path}")

    # Create deployer
    deployer = OllamaDeployer(
        gguf_path=gguf_path,
        model_name=args.model_name
    )

    # Execute deployment
    deployer.deploy(test=args.test)


if __name__ == '__main__':
    main()
