#!/usr/bin/env python3
"""
Convert Model to GGUF Format
Converts the merged model to GGUF format for use with llama.cpp and Ollama
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


class GGUFConverter:
    """Convert model to GGUF format"""

    def __init__(self, model_path, output_dir, llama_cpp_dir=None):
        """
        Initialize converter

        Args:
            model_path: Path to merged model
            output_dir: Output directory for GGUF files
            llama_cpp_dir: Path to llama.cpp (will clone if not provided)
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.llama_cpp_dir = Path(llama_cpp_dir) if llama_cpp_dir else Path.home() / 'llama.cpp'

    def check_model(self):
        """Check if model exists"""
        print("\n" + "="*60)
        print("Checking model...")
        print("="*60)

        if not self.model_path.exists():
            print(f"✗ Error: Model not found at {self.model_path}")
            print("  Please run model merging first")
            sys.exit(1)

        print(f"✓ Model found: {self.model_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output directory: {self.output_dir}")

    def setup_llama_cpp(self):
        """Setup llama.cpp repository"""
        print("\n" + "="*60)
        print("Setting up llama.cpp...")
        print("="*60)

        if self.llama_cpp_dir.exists():
            print(f"✓ llama.cpp found at: {self.llama_cpp_dir}")
            return True

        print(f"Cloning llama.cpp to: {self.llama_cpp_dir}")
        try:
            subprocess.run(
                ['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git',
                 str(self.llama_cpp_dir)],
                check=True,
                capture_output=True
            )
            print("✓ llama.cpp cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clone llama.cpp: {e}")
            sys.exit(1)

        # Install requirements
        print("Installing llama.cpp requirements...")
        requirements_file = self.llama_cpp_dir / 'requirements.txt'
        if requirements_file.exists():
            try:
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                    check=True,
                    capture_output=True
                )
                print("✓ Requirements installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Warning: Failed to install requirements: {e}")
                print("  Continuing anyway...")

        return True

    def convert_to_gguf(self):
        """Convert model to GGUF format"""
        print("\n" + "="*60)
        print("Converting to GGUF format...")
        print("="*60)

        convert_script = self.llama_cpp_dir / 'convert_hf_to_gguf.py'
        if not convert_script.exists():
            # Try alternate name
            convert_script = self.llama_cpp_dir / 'convert-hf-to-gguf.py'
            if not convert_script.exists():
                print(f"✗ Error: Conversion script not found in {self.llama_cpp_dir}")
                print("  Expected: convert_hf_to_gguf.py or convert-hf-to-gguf.py")
                sys.exit(1)

        output_file = self.output_dir / 'model-f16.gguf'

        print(f"Converting {self.model_path} to GGUF...")
        print("(This may take several minutes...)")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(convert_script),
                    str(self.model_path),
                    '--outfile', str(output_file),
                    '--outtype', 'f16'
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print("✓ Conversion complete")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ Conversion failed: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            sys.exit(1)

        if not output_file.exists():
            print(f"✗ Error: Output file not created: {output_file}")
            sys.exit(1)

        # Get file size
        size_gb = output_file.stat().st_size / (1024**3)
        print(f"✓ GGUF file created: {output_file} ({size_gb:.2f} GB)")

        return output_file

    def quantize_model(self, gguf_file, quant_type='Q5_K_M'):
        """
        Quantize GGUF model

        Args:
            gguf_file: Path to GGUF file
            quant_type: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)

        Returns:
            Path to quantized model
        """
        print("\n" + "="*60)
        print(f"Quantizing to {quant_type}...")
        print("="*60)

        # Build quantize binary if needed
        # Try both possible locations
        quantize_bin = self.llama_cpp_dir / 'build' / 'bin' / 'quantize'
        if not quantize_bin.exists():
            quantize_bin = self.llama_cpp_dir / 'quantize'

        if not quantize_bin.exists():
            print("Building llama.cpp with quantize binary...")
            build_dir = self.llama_cpp_dir / 'build'
            build_dir.mkdir(exist_ok=True)

            try:
                # Run cmake
                print("Running CMake configuration...")
                subprocess.run(
                    ['cmake', '..',
                     '-DCMAKE_BUILD_TYPE=Release',
                     '-DLLAMA_CURL=OFF',  # Disable CURL dependency
                     '-DLLAMA_BUILD_SERVER=OFF',  # We don't need the server
                     '-DLLAMA_BUILD_EXAMPLES=OFF'  # We don't need examples
                    ],
                    cwd=str(build_dir),
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✓ CMake configuration complete")

                # Build
                print("Building llama.cpp (this may take a few minutes)...")
                subprocess.run(
                    ['cmake', '--build', '.', '--config', 'Release'],
                    cwd=str(build_dir),
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("✓ Build complete")

                # Update quantize binary path
                quantize_bin = build_dir / 'bin' / 'quantize'
                if not quantize_bin.exists():
                    # Some systems put it directly in build/
                    quantize_bin = build_dir / 'quantize'

            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to build llama.cpp: {e}")
                if e.stderr:
                    print(f"  Error: {e.stderr[:500]}")
                print("  Skipping quantization")
                return None
            except FileNotFoundError:
                print("✗ CMake not found. Please install CMake:")
                print("  macOS: brew install cmake")
                print("  Linux: sudo apt-get install cmake")
                print("  Skipping quantization")
                return None

        if not quantize_bin.exists():
            print(f"✗ Quantize binary not found at {quantize_bin}")
            print("  Skipping quantization")
            return None

        # Quantize
        output_file = self.output_dir / f'model-{quant_type}.gguf'

        print(f"Quantizing to {quant_type}...")
        print("(This may take several minutes...)")

        try:
            subprocess.run(
                [
                    str(quantize_bin),
                    str(gguf_file),
                    str(output_file),
                    quant_type
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print("✓ Quantization complete")
        except subprocess.CalledProcessError as e:
            print(f"✗ Quantization failed: {e}")
            return None

        if not output_file.exists():
            print(f"✗ Error: Quantized file not created")
            return None

        # Get file size
        size_gb = output_file.stat().st_size / (1024**3)
        print(f"✓ Quantized file created: {output_file} ({size_gb:.2f} GB)")

        return output_file

    def convert(self, quantize=True, quant_type='Q5_K_M'):
        """
        Execute full conversion pipeline

        Args:
            quantize: Whether to quantize after conversion
            quant_type: Quantization type if quantizing
        """
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║   Converting Model to GGUF Format                        ║
        ╚══════════════════════════════════════════════════════════╝
        """)

        # Check model
        self.check_model()

        # Setup llama.cpp
        self.setup_llama_cpp()

        # Convert to GGUF
        gguf_file = self.convert_to_gguf()

        # Optionally quantize
        quantized_file = None
        if quantize:
            quantized_file = self.quantize_model(gguf_file, quant_type)

        print(f"""
        ╔══════════════════════════════════════════════════════════╗
        ║   ✓ Conversion Complete!                                 ║
        ║   GGUF file: {str(gguf_file.name):43s} ║""")

        if quantized_file:
            print(f"║   Quantized: {str(quantized_file.name):43s} ║")

        print(f"""║   Location:  {str(self.output_dir):43s} ║
        ╚══════════════════════════════════════════════════════════╝
        """)

        return quantized_file if quantized_file else gguf_file


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert model to GGUF format'
    )
    parser.add_argument('--model_path', type=str,
                       default='ORLM/checkpoints/orlm-qwen3-8b-merged',
                       help='Path to merged model')
    parser.add_argument('--output_dir', type=str,
                       default='ORLM/checkpoints/gguf',
                       help='Output directory for GGUF files')
    parser.add_argument('--llama_cpp_dir', type=str,
                       default=None,
                       help='Path to llama.cpp (will clone if not provided)')
    parser.add_argument('--quantize', action='store_true', default=True,
                       help='Quantize the model (default: True)')
    parser.add_argument('--no-quantize', dest='quantize', action='store_false',
                       help='Skip quantization')
    parser.add_argument('--quant_type', type=str,
                       default='Q5_K_M',
                       choices=['Q4_0', 'Q4_K_M', 'Q5_0', 'Q5_K_M', 'Q8_0'],
                       help='Quantization type')

    args = parser.parse_args()

    # Create converter
    converter = GGUFConverter(
        model_path=args.model_path,
        output_dir=args.output_dir,
        llama_cpp_dir=args.llama_cpp_dir
    )

    # Execute conversion
    converter.convert(
        quantize=args.quantize,
        quant_type=args.quant_type
    )


if __name__ == '__main__':
    main()
