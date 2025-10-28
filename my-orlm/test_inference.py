#!/usr/bin/env python3
"""
Test Inference Script
Test the fine-tuned model with sample operations research problems
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


class InferenceTester:
    """Test inference on fine-tuned model"""

    def __init__(self, model_path, device='auto'):
        """
        Initialize tester

        Args:
            model_path: Path to merged model
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer"""
        print("\n" + "="*60)
        print(f"Loading model from: {self.model_path}")
        print("="*60)

        if not self.model_path.exists():
            print(f"✗ Error: Model not found at {self.model_path}")
            print("  Please run model merging first")
            sys.exit(1)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print(f"✓ Model loaded")

        # Check device
        if torch.cuda.is_available():
            print(f"✓ Running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ Running on CPU (slower)")

    def generate(self, prompt, max_new_tokens=512, temperature=0.2, do_sample=True):
        """
        Generate response for a prompt

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def test_samples(self):
        """Test with sample OR problems"""
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║   Testing Model with Sample OR Problems                 ║
        ╚══════════════════════════════════════════════════════════╝
        """)

        # Sample problems
        test_problems = [
            {
                "title": "Transportation Problem",
                "prompt": """A company has 3 warehouses and 4 retail stores. The supply at warehouses is [100, 150, 200] units, and demand at stores is [80, 90, 110, 120] units. The transportation cost matrix is:
    Store1  Store2  Store3  Store4
W1:   8      6      10      9
W2:   9      12     13      7
W3:   14     9      16      5

Formulate this as a linear programming problem to minimize transportation costs."""
            },
            {
                "title": "Production Planning",
                "prompt": """A factory produces two products A and B. Product A requires 2 hours of machine time and 3 hours of labor, yielding profit $50. Product B requires 3 hours of machine time and 2 hours of labor, yielding profit $60. The factory has 100 hours of machine time and 90 hours of labor available per day. Formulate an optimization problem to maximize profit."""
            },
            {
                "title": "Assignment Problem",
                "prompt": """There are 4 tasks and 4 workers. The cost matrix for assigning worker i to task j is:
    Task1  Task2  Task3  Task4
W1:  15     10     9      8
W2:  9      15     8      7
W3:  6      12     7      6
W4:  10     8      11     9

Formulate this as an assignment problem to minimize total cost."""
            }
        ]

        results = []

        for i, problem in enumerate(test_problems, 1):
            print(f"\n{'='*60}")
            print(f"Test Case {i}: {problem['title']}")
            print('='*60)
            print(f"\nPrompt:\n{problem['prompt']}\n")
            print("-" * 60)
            print("Generating response...")
            print("-" * 60)

            # Generate response
            response = self.generate(
                problem['prompt'],
                max_new_tokens=800,
                temperature=0.2
            )

            print(f"\nResponse:\n{response}\n")

            results.append({
                'problem': problem['title'],
                'prompt': problem['prompt'],
                'response': response
            })

            print("="*60)
            input("\nPress Enter to continue to next test case...")

        return results

    def interactive_mode(self):
        """Interactive testing mode"""
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║   Interactive Testing Mode                               ║
        ║   Enter your OR problems (type 'exit' to quit)           ║
        ╚══════════════════════════════════════════════════════════╝
        """)

        while True:
            print("\n" + "="*60)
            prompt = input("Enter your optimization problem (or 'exit'): ").strip()

            if prompt.lower() in ['exit', 'quit', 'q']:
                print("Exiting interactive mode...")
                break

            if not prompt:
                continue

            print("\n" + "-"*60)
            print("Generating response...")
            print("-"*60)

            response = self.generate(prompt, max_new_tokens=800)
            print(f"\nResponse:\n{response}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test inference on fine-tuned ORLM model'
    )
    parser.add_argument('--model_path', type=str,
                       default='ORLM/checkpoints/orlm-qwen3-8b-merged',
                       help='Path to merged model')
    parser.add_argument('--mode', type=str,
                       choices=['sample', 'interactive', 'both'],
                       default='sample',
                       help='Testing mode')
    parser.add_argument('--device', type=str,
                       default='auto',
                       help='Device to use (auto, cuda, cpu)')

    args = parser.parse_args()

    # Create tester
    tester = InferenceTester(
        model_path=args.model_path,
        device=args.device
    )

    # Load model
    tester.load_model()

    # Run tests
    if args.mode in ['sample', 'both']:
        tester.test_samples()

    if args.mode in ['interactive', 'both']:
        tester.interactive_mode()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   ✓ Testing Complete!                                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
