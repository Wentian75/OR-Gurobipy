#!/usr/bin/env python3
"""
QLoRA Fine-tuning Script for ORLM-Qwen3-8B
Automated training with configurable parameters
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch


class QLorTrainer:
    """QLoRA Training Pipeline"""

    def __init__(self, config=None):
        """Initialize trainer with configuration"""
        self.config = config or self.get_default_config()
        self.setup_environment()

    @staticmethod
    def get_default_config():
        """Get default training configuration"""
        return {
            'model_name': 'Qwen/Qwen2.5-7B-Instruct',
            'data_path': 'ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl',
            'output_dir': 'ORLM/checkpoints/orlm-qwen3-8b-qlora',
            'seq_len': 2048,
            'batch_size': 1,
            'grad_acc': 8,
            'epochs': 1,
            'learning_rate': 2e-4,
            'lora_r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0.05,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'gate_proj', 'up_proj', 'down_proj'],
            'warmup_ratio': 0.03,
            'logging_steps': 10,
            'save_strategy': 'epoch',
            'save_total_limit': 2,
        }

    def setup_environment(self):
        """Setup environment variables"""
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_SILENT'] = 'true'

    def check_requirements(self):
        """Check if all requirements are met"""
        print("\n" + "="*60)
        print("Checking requirements...")
        print("="*60)

        # Check CUDA
        if not torch.cuda.is_available():
            print("⚠ Warning: CUDA not available. Training will be slow on CPU.")
        else:
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

        # Check data file
        data_path = Path(self.config['data_path'])
        if not data_path.exists():
            print(f"✗ Error: Data file not found at {data_path}")
            print(f"  Please ensure the training data exists at: {data_path}")
            sys.exit(1)
        else:
            print(f"✓ Training data found: {data_path}")
            # Count lines
            line_count = sum(1 for _ in open(data_path, 'r', encoding='utf-8'))
            print(f"  Dataset size: {line_count} examples")

        # Check output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {output_dir}")

    def build_dataset(self, data_path, tokenizer, max_len):
        """Build and tokenize dataset"""
        print("\n" + "="*60)
        print("Loading and tokenizing dataset...")
        print("="*60)

        # Load dataset
        raw_dataset = load_dataset('json', data_files=data_path, split='train')
        print(f"✓ Loaded {len(raw_dataset)} examples")

        # Format dataset
        def format_text(example):
            prompt = (example.get('prompt') or '').strip()
            completion = (example.get('completion') or '').strip()
            return {'text': f"{prompt}\n{completion}"}

        raw_dataset = raw_dataset.map(
            format_text,
            remove_columns=raw_dataset.column_names
        )

        # Tokenize
        def tokenize_function(batch):
            return tokenizer(
                batch['text'],
                truncation=True,
                max_length=max_len,
                padding='max_length',
            )

        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            desc="Tokenizing dataset"
        )

        print(f"✓ Dataset tokenized successfully")
        return tokenized_dataset

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with QLoRA config"""
        print("\n" + "="*60)
        print(f"Loading model: {self.config['model_name']}")
        print("="*60)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            use_fast=True,
            trust_remote_code=True
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✓ Tokenizer loaded")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True
        )

        print(f"✓ Model loaded with 4-bit quantization")

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=self.config['target_modules']
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ LoRA applied")
        print(f"  Trainable params: {trainable_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
        print(f"  Total params: {total_params:,}")

        return model, tokenizer

    def train(self):
        """Execute training pipeline"""
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║   Starting QLoRA Fine-tuning Pipeline                    ║
        ╚══════════════════════════════════════════════════════════╝
        """)

        # Check requirements
        self.check_requirements()

        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Build dataset
        train_dataset = self.build_dataset(
            self.config['data_path'],
            tokenizer,
            self.config['seq_len']
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['grad_acc'],
            num_train_epochs=self.config['epochs'],
            learning_rate=self.config['learning_rate'],
            lr_scheduler_type='linear',
            warmup_ratio=self.config['warmup_ratio'],
            logging_steps=self.config['logging_steps'],
            save_strategy=self.config['save_strategy'],
            save_total_limit=self.config['save_total_limit'],
            fp16=True,
            gradient_checkpointing=True,
            optim='paged_adamw_8bit',
            dataloader_num_workers=4,
            report_to='none',
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Print training info
        print("\n" + "="*60)
        print("Training Configuration:")
        print("="*60)
        print(f"  Model: {self.config['model_name']}")
        print(f"  Dataset: {self.config['data_path']}")
        print(f"  Output: {self.config['output_dir']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Gradient accumulation: {self.config['grad_acc']}")
        print(f"  Effective batch size: {self.config['batch_size'] * self.config['grad_acc']}")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  LoRA rank: {self.config['lora_r']}")
        print(f"  LoRA alpha: {self.config['lora_alpha']}")
        print("="*60)

        # Start training
        print("\n🚀 Starting training...\n")
        trainer.train()

        # Save final model
        print("\n" + "="*60)
        print("Saving model...")
        print("="*60)
        trainer.model.save_pretrained(self.config['output_dir'])
        tokenizer.save_pretrained(self.config['output_dir'])

        print(f"""
        ╔══════════════════════════════════════════════════════════╗
        ║   ✓ Training Complete!                                   ║
        ║   Model saved to: {self.config['output_dir']:36s} ║
        ╚══════════════════════════════════════════════════════════╝
        """)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='QLoRA Fine-tuning for ORLM-Qwen3-8B'
    )
    parser.add_argument('--model_name', type=str,
                       default='Qwen/Qwen2.5-7B-Instruct',
                       help='Base model name or path')
    parser.add_argument('--data_path', type=str,
                       default='ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl',
                       help='Path to training data (JSONL format)')
    parser.add_argument('--output_dir', type=str,
                       default='ORLM/checkpoints/orlm-qwen3-8b-qlora',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Per-device batch size')
    parser.add_argument('--grad_acc', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=float, default=1,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=64,
                       help='LoRA rank')

    args = parser.parse_args()

    # Build config from args
    config = QLorTrainer.get_default_config()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # Create trainer and start training
    trainer = QLorTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
