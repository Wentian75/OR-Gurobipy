#!/usr/bin/env python3
"""
Merge LoRA Adapter with Base Model
Combines the QLoRA adapter weights with the base model to create a full model
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ModelMerger:
    """Merge LoRA adapters with base model"""

    def __init__(self, base_model, adapter_path, output_path):
        """
        Initialize merger

        Args:
            base_model: Base model name or path
            adapter_path: Path to LoRA adapter
            output_path: Output path for merged model
        """
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.output_path = Path(output_path)

    def check_paths(self):
        """Verify input paths exist"""
        print("\n" + "="*60)
        print("Checking paths...")
        print("="*60)

        if not self.adapter_path.exists():
            print(f"✗ Error: Adapter path not found: {self.adapter_path}")
            print("  Please ensure training completed successfully")
            sys.exit(1)
        else:
            print(f"✓ Adapter found: {self.adapter_path}")

        # Check if adapter has required files
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        missing_files = []
        for file in required_files:
            if not (self.adapter_path / file).exists():
                # Try .bin format
                if file == 'adapter_model.safetensors':
                    if not (self.adapter_path / 'adapter_model.bin').exists():
                        missing_files.append(file)
                else:
                    missing_files.append(file)

        if missing_files:
            print(f"⚠ Warning: Some adapter files may be missing: {missing_files}")
        else:
            print(f"✓ All required adapter files found")

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {self.output_path}")

    def merge(self):
        """Merge LoRA adapter with base model"""
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║   Merging LoRA Adapter with Base Model                  ║
        ╚══════════════════════════════════════════════════════════╝
        """)

        # Check paths
        self.check_paths()

        # Load tokenizer
        print("\n" + "="*60)
        print("Loading tokenizer...")
        print("="*60)
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded from {self.base_model}")

        # Load base model
        print("\n" + "="*60)
        print(f"Loading base model: {self.base_model}")
        print("="*60)
        print("(This may take several minutes...)")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"✓ Base model loaded")

        # Load and merge adapter
        print("\n" + "="*60)
        print("Loading LoRA adapter...")
        print("="*60)

        model = PeftModel.from_pretrained(
            base_model,
            str(self.adapter_path)
        )
        print(f"✓ Adapter loaded from {self.adapter_path}")

        print("\n" + "="*60)
        print("Merging adapter with base model...")
        print("="*60)
        print("(This may take several minutes...)")

        model = model.merge_and_unload()
        print(f"✓ Merge complete")

        # Save merged model
        print("\n" + "="*60)
        print("Saving merged model...")
        print("="*60)

        tokenizer.save_pretrained(str(self.output_path))
        print(f"✓ Tokenizer saved")

        model.save_pretrained(
            str(self.output_path),
            safe_serialization=True,
            max_shard_size="5GB"
        )
        print(f"✓ Model saved")

        # Get model size
        total_size = sum(
            f.stat().st_size for f in self.output_path.glob('**/*') if f.is_file()
        )
        size_gb = total_size / (1024**3)

        print(f"""
        ╔══════════════════════════════════════════════════════════╗
        ║   ✓ Merge Complete!                                      ║
        ║   Merged model saved to:                                 ║
        ║   {str(self.output_path):54s} ║
        ║   Size: {size_gb:6.2f} GB                                       ║
        ╚══════════════════════════════════════════════════════════╝
        """)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge LoRA adapter with base model'
    )
    parser.add_argument('--base_model', type=str,
                       default='Qwen/Qwen3-8B',
                       help='Base model name or path')
    parser.add_argument('--adapter_path', type=str,
                       default='ORLM/checkpoints/orlm-qwen3-8b-qlora',
                       help='Path to LoRA adapter')
    parser.add_argument('--output_path', type=str,
                       default='ORLM/checkpoints/orlm-qwen3-8b-merged',
                       help='Output path for merged model')

    args = parser.parse_args()

    # Create merger and execute
    merger = ModelMerger(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path
    )
    merger.merge()


if __name__ == '__main__':
    main()
