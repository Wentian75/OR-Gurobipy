#!/usr/bin/env python3
"""
Complete Training Pipeline Orchestrator
Runs the entire workflow from setup to deployment
"""

import sys
import argparse
from pathlib import Path

# Import our modules
try:
    from setup_environment import main as setup_env
    from train_qlora import QLorTrainer
    from merge_model import ModelMerger
    from test_inference import InferenceTester
    from convert_to_gguf import GGUFConverter
    from deploy_ollama import OllamaDeployer
except ImportError:
    print("Error: Could not import required modules.")
    print("Make sure all scripts are in the same directory.")
    sys.exit(1)


class TrainingPipeline:
    """Orchestrate the complete training pipeline"""

    def __init__(self, config=None):
        """Initialize pipeline with configuration"""
        self.config = config or self.get_default_config()

    @staticmethod
    def get_default_config():
        """Get default pipeline configuration"""
        return {
            # Model and data
            'base_model': 'Qwen/Qwen2.5-7B-Instruct',
            'data_path': 'ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl',

            # Training
            'adapter_output': 'ORLM/checkpoints/orlm-qwen3-8b-qlora',
            'batch_size': 1,
            'grad_acc': 8,
            'epochs': 1,
            'learning_rate': 2e-4,

            # Merging
            'merged_output': 'ORLM/checkpoints/orlm-qwen3-8b-merged',

            # GGUF conversion
            'gguf_output': 'ORLM/checkpoints/gguf',
            'quantize': True,
            'quant_type': 'Q5_K_M',

            # Ollama
            'ollama_model_name': 'orlm-qwen3-8b',

            # Pipeline control
            'skip_setup': False,
            'skip_training': False,
            'skip_merge': False,
            'skip_test': False,
            'skip_gguf': False,
            'skip_ollama': False,
        }

    def print_banner(self):
        """Print pipeline banner"""
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║                                                          ║
        ║     ORLM-Qwen3 QLoRA Training Pipeline                   ║
        ║     Automated Fine-tuning and Deployment                 ║
        ║                                                          ║
        ╚══════════════════════════════════════════════════════════╝
        """)

    def print_config(self):
        """Print pipeline configuration"""
        print("\n" + "="*60)
        print("Pipeline Configuration:")
        print("="*60)
        print(f"Base model:      {self.config['base_model']}")
        print(f"Training data:   {self.config['data_path']}")
        print(f"Adapter output:  {self.config['adapter_output']}")
        print(f"Merged output:   {self.config['merged_output']}")
        print(f"GGUF output:     {self.config['gguf_output']}")
        print(f"Ollama name:     {self.config['ollama_model_name']}")
        print("="*60)

        # Check what will be skipped
        skipped = [k.replace('skip_', '').upper()
                  for k, v in self.config.items()
                  if k.startswith('skip_') and v]
        if skipped:
            print(f"\nSkipping stages: {', '.join(skipped)}")
            print("="*60)

    def run_setup(self):
        """Step 1: Setup environment"""
        if self.config['skip_setup']:
            print("\n⏭️  Skipping environment setup")
            return

        print("\n" + "="*60)
        print("STEP 1: Environment Setup")
        print("="*60)

        try:
            setup_env()
            print("✓ Environment setup complete")
        except Exception as e:
            print(f"✗ Setup failed: {e}")
            return False

        return True

    def run_training(self):
        """Step 2: QLoRA training"""
        if self.config['skip_training']:
            print("\n⏭️  Skipping training")
            return True

        print("\n" + "="*60)
        print("STEP 2: QLoRA Fine-tuning")
        print("="*60)

        try:
            trainer_config = {
                'model_name': self.config['base_model'],
                'data_path': self.config['data_path'],
                'output_dir': self.config['adapter_output'],
                'batch_size': self.config['batch_size'],
                'grad_acc': self.config['grad_acc'],
                'epochs': self.config['epochs'],
                'learning_rate': self.config['learning_rate'],
            }
            trainer_config.update(QLorTrainer.get_default_config())
            trainer_config.update({
                'model_name': self.config['base_model'],
                'data_path': self.config['data_path'],
                'output_dir': self.config['adapter_output'],
            })

            trainer = QLorTrainer(trainer_config)
            trainer.train()
            print("✓ Training complete")
            return True

        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_merge(self):
        """Step 3: Merge LoRA adapter"""
        if self.config['skip_merge']:
            print("\n⏭️  Skipping model merge")
            return True

        print("\n" + "="*60)
        print("STEP 3: Merge LoRA Adapter")
        print("="*60)

        try:
            merger = ModelMerger(
                base_model=self.config['base_model'],
                adapter_path=self.config['adapter_output'],
                output_path=self.config['merged_output']
            )
            merger.merge()
            print("✓ Merge complete")
            return True

        except Exception as e:
            print(f"✗ Merge failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_test(self):
        """Step 4: Test inference"""
        if self.config['skip_test']:
            print("\n⏭️  Skipping inference test")
            return True

        print("\n" + "="*60)
        print("STEP 4: Test Inference")
        print("="*60)

        try:
            tester = InferenceTester(
                model_path=self.config['merged_output']
            )
            tester.load_model()
            print("\n✓ Model loaded successfully")
            print("  (Skipping sample tests in automated pipeline)")
            print("  Run 'python test_inference.py' for interactive testing")
            return True

        except Exception as e:
            print(f"✗ Inference test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_gguf_conversion(self):
        """Step 5: Convert to GGUF"""
        if self.config['skip_gguf']:
            print("\n⏭️  Skipping GGUF conversion")
            return True

        print("\n" + "="*60)
        print("STEP 5: Convert to GGUF")
        print("="*60)

        try:
            converter = GGUFConverter(
                model_path=self.config['merged_output'],
                output_dir=self.config['gguf_output']
            )
            gguf_file = converter.convert(
                quantize=self.config['quantize'],
                quant_type=self.config['quant_type']
            )
            self.config['gguf_file'] = str(gguf_file)
            print("✓ GGUF conversion complete")
            return True

        except Exception as e:
            print(f"✗ GGUF conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_ollama_deployment(self):
        """Step 6: Deploy to Ollama"""
        if self.config['skip_ollama']:
            print("\n⏭️  Skipping Ollama deployment")
            return True

        print("\n" + "="*60)
        print("STEP 6: Deploy to Ollama")
        print("="*60)

        try:
            # Find GGUF file
            gguf_file = self.config.get('gguf_file')
            if not gguf_file:
                # Try to find it
                gguf_dir = Path(self.config['gguf_output'])
                gguf_files = list(gguf_dir.glob('model-Q*.gguf'))
                if not gguf_files:
                    gguf_files = list(gguf_dir.glob('*.gguf'))
                if not gguf_files:
                    print("✗ No GGUF file found")
                    return False
                gguf_file = str(gguf_files[0])

            deployer = OllamaDeployer(
                gguf_path=gguf_file,
                model_name=self.config['ollama_model_name']
            )
            deployer.deploy(test=False)  # Skip test in automated pipeline
            print("✓ Ollama deployment complete")
            return True

        except Exception as e:
            print(f"✗ Ollama deployment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run(self):
        """Execute complete pipeline"""
        self.print_banner()
        self.print_config()

        print("\n" + "="*60)
        print("Starting pipeline...")
        print("="*60)

        # Pipeline steps
        steps = [
            ("Setup Environment", self.run_setup),
            ("QLoRA Training", self.run_training),
            ("Merge Model", self.run_merge),
            ("Test Inference", self.run_test),
            ("GGUF Conversion", self.run_gguf_conversion),
            ("Ollama Deployment", self.run_ollama_deployment),
        ]

        completed_steps = []
        failed_step = None

        for step_name, step_func in steps:
            try:
                result = step_func()
                if result is False:
                    failed_step = step_name
                    break
                completed_steps.append(step_name)
            except KeyboardInterrupt:
                print("\n\n⚠️  Pipeline interrupted by user")
                failed_step = step_name
                break
            except Exception as e:
                print(f"\n✗ Unexpected error in {step_name}: {e}")
                import traceback
                traceback.print_exc()
                failed_step = step_name
                break

        # Print summary
        self.print_summary(completed_steps, failed_step)

    def print_summary(self, completed_steps, failed_step):
        """Print pipeline summary"""
        print("\n\n" + "="*60)
        print("Pipeline Summary")
        print("="*60)

        if completed_steps:
            print("\n✓ Completed steps:")
            for step in completed_steps:
                print(f"  • {step}")

        if failed_step:
            print(f"\n✗ Failed at: {failed_step}")
            print("\nYou can resume from this step or run individual scripts:")
            print("  - setup_environment.py")
            print("  - train_qlora.py")
            print("  - merge_model.py")
            print("  - test_inference.py")
            print("  - convert_to_gguf.py")
            print("  - deploy_ollama.py")
        else:
            print(f"""
        ╔══════════════════════════════════════════════════════════╗
        ║   🎉 Pipeline Complete!                                  ║
        ║                                                          ║
        ║   Your model is ready to use:                            ║
        ║   $ ollama run {self.config['ollama_model_name']:43s} ║
        ╚══════════════════════════════════════════════════════════╝
            """)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Complete ORLM-Qwen3 QLoRA training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_training_pipeline.py

  # Skip setup (already done)
  python run_training_pipeline.py --skip-setup

  # Quick test (skip GGUF and Ollama)
  python run_training_pipeline.py --skip-gguf --skip-ollama

  # Resume from merge (training already done)
  python run_training_pipeline.py --skip-setup --skip-training
        """
    )

    # Model and data
    parser.add_argument('--base_model', type=str,
                       default='Qwen/Qwen2.5-7B-Instruct',
                       help='Base model name')
    parser.add_argument('--data_path', type=str,
                       default='ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl',
                       help='Training data path')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_acc', type=int, default=8)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)

    # Output paths
    parser.add_argument('--adapter_output', type=str,
                       default='ORLM/checkpoints/orlm-qwen3-8b-qlora')
    parser.add_argument('--merged_output', type=str,
                       default='ORLM/checkpoints/orlm-qwen3-8b-merged')
    parser.add_argument('--gguf_output', type=str,
                       default='ORLM/checkpoints/gguf')

    # GGUF options
    parser.add_argument('--quantize', action='store_true', default=True)
    parser.add_argument('--no-quantize', dest='quantize', action='store_false')
    parser.add_argument('--quant_type', type=str, default='Q5_K_M',
                       choices=['Q4_0', 'Q4_K_M', 'Q5_0', 'Q5_K_M', 'Q8_0'])

    # Ollama
    parser.add_argument('--ollama_model_name', type=str,
                       default='orlm-qwen3-8b')

    # Pipeline control
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip environment setup')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing adapter)')
    parser.add_argument('--skip-merge', action='store_true',
                       help='Skip merging (use existing merged model)')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip inference testing')
    parser.add_argument('--skip-gguf', action='store_true',
                       help='Skip GGUF conversion')
    parser.add_argument('--skip-ollama', action='store_true',
                       help='Skip Ollama deployment')

    args = parser.parse_args()

    # Build config
    config = TrainingPipeline.get_default_config()
    for key, value in vars(args).items():
        config[key] = value

    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
