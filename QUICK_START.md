# ORLM-Qwen3-8B: Automated QLoRA Fine-tuning

**Simple, automated fine-tuning pipeline for operations research language models**

This repository provides a fully automated Python-based pipeline for fine-tuning Qwen3-8B on operations research problems using QLoRA (4-bit quantized LoRA).

---

## 📋 Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with at least 24GB VRAM (recommended)
- **Disk Space**: ~50GB free space
- **RAM**: 32GB+ recommended
- **OS**: Linux (Ubuntu/CentOS) or macOS

### Software Requirements
- Python 3.8+
- CUDA 12.1+ (for GPU training)
- Git

---

## 🚀 Quick Start (3 Commands!)

### Option 1: Complete Automated Pipeline

```bash
# 1. Activate your Python virtual environment
cd ~/work/OR  # or wherever you cloned the repo
source venv/bin/activate

# 2. Run the complete pipeline (this does everything!)
python my-orlm/run_training_pipeline.py
```

That's it! The pipeline will:
1. ✅ Install all dependencies
2. ✅ Fine-tune the model with QLoRA
3. ✅ Merge LoRA adapters with base model
4. ✅ Test the model
5. ✅ Convert to GGUF format
6. ✅ Deploy to Ollama

**Time estimate**: 2-4 hours (depending on your GPU)

### Option 2: Step-by-Step Execution

If you prefer to run each step individually:

```bash
# Step 1: Setup environment (one-time)
python my-orlm/setup_environment.py

# Step 2: Train with QLoRA
python my-orlm/train_qlora.py

# Step 3: Merge LoRA adapter with base model
python my-orlm/merge_model.py

# Step 4: Test the model
python my-orlm/test_inference.py --mode sample

# Step 5: Convert to GGUF (for Ollama)
python my-orlm/convert_to_gguf.py

# Step 6: Deploy to Ollama
python my-orlm/deploy_ollama.py
```

---

## 📁 Project Structure

```
OR/
├── my-orlm/                    # Automated training scripts
│   ├── run_training_pipeline.py   # 🌟 Main orchestrator (run this!)
│   ├── setup_environment.py       # Environment setup
│   ├── train_qlora.py            # QLoRA training
│   ├── merge_model.py            # Merge LoRA adapters
│   ├── test_inference.py         # Test model
│   ├── convert_to_gguf.py        # Convert to GGUF
│   └── deploy_ollama.py          # Deploy to Ollama
│
├── ORLM/
│   ├── data/                      # Training data
│   │   └── OR-Instruct-Data-3K-Gurobipy.jsonl  # Your training dataset
│   └── checkpoints/               # Model checkpoints (created during training)
│       ├── orlm-qwen3-8b-qlora/  # LoRA adapters
│       ├── orlm-qwen3-8b-merged/ # Merged model
│       └── gguf/                 # GGUF files
│
└── venv/                          # Python virtual environment
```

---

## 🎯 Usage Examples

### Basic Training

```bash
# Run complete pipeline with defaults
python my-orlm/run_training_pipeline.py
```

### Custom Parameters

```bash
# Custom training parameters
python my-orlm/run_training_pipeline.py \
    --base_model "Qwen/Qwen3-8B" \
    --data_path "ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl" \
    --epochs 2 \
    --batch_size 2 \
    --learning_rate 1e-4
```

### Resume from Checkpoint

```bash
# Skip setup and training if already done
python my-orlm/run_training_pipeline.py \
    --skip-setup \
    --skip-training
```

### Quick Test (Skip Deployment)

```bash
# Skip GGUF conversion and Ollama deployment
python my-orlm/run_training_pipeline.py \
    --skip-gguf \
    --skip-ollama
```

---

## 🛠️ Individual Script Usage

### 1. Setup Environment

```bash
python my-orlm/setup_environment.py
```

**What it does:**
- Installs PyTorch with CUDA support
- Installs QLoRA dependencies (transformers, peft, bitsandbytes, etc.)
- Configures environment variables
- Verifies GPU availability

### 2. Train with QLoRA

```bash
# Basic training
python my-orlm/train_qlora.py

# With custom parameters
python my-orlm/train_qlora.py \
    --model_name "Qwen/Qwen3-8B" \
    --data_path "ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl" \
    --output_dir "ORLM/checkpoints/orlm-qwen3-8b-qlora" \
    --epochs 1 \
    --batch_size 1 \
    --grad_acc 8 \
    --learning_rate 2e-4 \
    --lora_r 64
```

**Training parameters:**
- `--batch_size`: Per-device batch size (default: 1)
- `--grad_acc`: Gradient accumulation steps (default: 8)
  - Effective batch size = batch_size × grad_acc = 8
- `--epochs`: Number of training epochs (default: 1)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--lora_r`: LoRA rank (default: 64)

**Output:**
- LoRA adapter saved to `ORLM/checkpoints/orlm-qwen3-8b-qlora/`
- Training log saved automatically

### 3. Merge Model

```bash
# Basic merge
python my-orlm/merge_model.py

# With custom paths
python my-orlm/merge_model.py \
    --base_model "Qwen/Qwen3-8B" \
    --adapter_path "ORLM/checkpoints/orlm-qwen3-8b-qlora" \
    --output_path "ORLM/checkpoints/orlm-qwen3-8b-merged"
```

**What it does:**
- Loads base model and LoRA adapter
- Merges adapter weights into base model
- Saves full merged model

**Output:**
- Merged model saved to `ORLM/checkpoints/orlm-qwen3-8b-merged/`

### 4. Test Inference

```bash
# Test with sample problems
python my-orlm/test_inference.py --mode sample

# Interactive testing
python my-orlm/test_inference.py --mode interactive

# Both sample and interactive
python my-orlm/test_inference.py --mode both

# Custom model path
python my-orlm/test_inference.py \
    --model_path "ORLM/checkpoints/orlm-qwen3-8b-merged" \
    --mode interactive
```

**Testing modes:**
- `sample`: Test with 3 pre-defined OR problems
- `interactive`: Enter your own problems
- `both`: Run sample tests then interactive mode

### 5. Convert to GGUF

```bash
# Basic conversion with quantization
python my-orlm/convert_to_gguf.py

# Without quantization
python my-orlm/convert_to_gguf.py --no-quantize

# Custom quantization type
python my-orlm/convert_to_gguf.py --quant_type Q4_K_M

# Custom paths
python my-orlm/convert_to_gguf.py \
    --model_path "ORLM/checkpoints/orlm-qwen3-8b-merged" \
    --output_dir "ORLM/checkpoints/gguf" \
    --quant_type Q5_K_M
```

**Quantization types:**
- `Q4_0`: 4-bit quantization (smallest, fastest)
- `Q4_K_M`: 4-bit with K-quants (good balance)
- `Q5_0`: 5-bit quantization
- `Q5_K_M`: 5-bit with K-quants (recommended)
- `Q8_0`: 8-bit quantization (largest, highest quality)

**Output:**
- FP16 GGUF: `ORLM/checkpoints/gguf/model-f16.gguf`
- Quantized: `ORLM/checkpoints/gguf/model-Q5_K_M.gguf`

### 6. Deploy to Ollama

```bash
# Basic deployment
python my-orlm/deploy_ollama.py

# Custom model name
python my-orlm/deploy_ollama.py --model_name my-or-model

# Skip testing
python my-orlm/deploy_ollama.py --no-test

# Custom GGUF file
python my-orlm/deploy_ollama.py \
    --gguf_path "ORLM/checkpoints/gguf/model-Q5_K_M.gguf" \
    --model_name "orlm-qwen3-8b"
```

**What it does:**
- Checks Ollama installation
- Creates Modelfile with OR-specific prompts
- Registers model with Ollama
- Optionally tests the deployed model

**Using the deployed model:**
```bash
# Run the model
ollama run orlm-qwen3-8b

# Example prompt
ollama run orlm-qwen3-8b "A factory produces products A and B. Product A yields $50 profit and requires 2 hours. Product B yields $60 profit and requires 3 hours. There are 100 hours available. Formulate an optimization problem to maximize profit."

# Remove the model
ollama rm orlm-qwen3-8b
```

---

## 📊 Expected Results

### Training Metrics
- **Training time**: 1-3 hours (depending on GPU)
- **GPU memory**: ~20GB VRAM
- **Training loss**: Should decrease from ~2.0 to ~0.5
- **Trainable parameters**: ~200M (2.5% of total)

### Model Sizes
- **LoRA adapter**: ~400MB
- **Merged model**: ~15GB (FP16)
- **GGUF (Q5_K_M)**: ~5-6GB
- **GGUF (Q4_K_M)**: ~4-5GB

### Inference Performance
- **Generation speed**: ~20-30 tokens/sec (on A100)
- **Response length**: 200-800 tokens (typical OR problem)

---

## ⚙️ Configuration Options

### Pipeline Configuration

All scripts accept command-line arguments for customization:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base_model` | `Qwen/Qwen3-8B` | Base model to fine-tune |
| `--data_path` | `ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl` | Training data |
| `--batch_size` | `1` | Per-device batch size |
| `--grad_acc` | `8` | Gradient accumulation steps |
| `--epochs` | `1` | Number of training epochs |
| `--learning_rate` | `2e-4` | Learning rate |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha |
| `--quant_type` | `Q5_K_M` | GGUF quantization type |

### Environment Variables

The scripts automatically set these, but you can override:

```bash
export CUDA_VISIBLE_DEVICES=0          # Select GPU
export TOKENIZERS_PARALLELISM=false    # Disable tokenizer warnings
export WANDB_DISABLED=true             # Disable W&B logging
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size or increase gradient accumulation
python my-orlm/train_qlora.py --batch_size 1 --grad_acc 16

# Or reduce sequence length (edit in train_qlora.py)
# Change SEQ_LEN from 2048 to 1024
```

### Issue: CUDA Not Available

**Check CUDA:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution:**
- Reinstall PyTorch with correct CUDA version
- Check NVIDIA driver installation

### Issue: Ollama Not Found

**Install Ollama:**
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.com/download
```

### Issue: llama.cpp Compilation Failed

**Solution:**
```bash
# Install build tools
sudo apt-get install build-essential cmake  # Ubuntu
brew install cmake  # macOS

# Manually build llama.cpp
cd ~/llama.cpp
make quantize
```

### Issue: Training Loss Not Decreasing

**Solutions:**
- Increase learning rate: `--learning_rate 5e-4`
- Increase LoRA rank: `--lora_r 128`
- Train for more epochs: `--epochs 2`
- Check data quality

### Issue: Model Generates Gibberish

**Solutions:**
- Lower temperature in generation
- Check if training completed successfully
- Verify merged model contains both base and adapter
- Try generating with more structured prompts

---

## 📚 Additional Resources

### Training Data Format

Your training data should be in JSONL format:

```json
{"prompt": "A company has...", "completion": "Here's the optimization model..."}
{"prompt": "Minimize cost...", "completion": "We can formulate this as..."}
```

### Custom Training Data

To use your own data:

```bash
# 1. Prepare your data in JSONL format
# 2. Save to ORLM/data/my_data.jsonl
# 3. Run training with custom data
python my-orlm/train_qlora.py --data_path "ORLM/data/my_data.jsonl"
```

### Model Evaluation

After training, evaluate on your test set:

```python
from test_inference import InferenceTester

tester = InferenceTester("ORLM/checkpoints/orlm-qwen3-8b-merged")
tester.load_model()

# Test on your examples
response = tester.generate("Your OR problem here")
print(response)
```

---

## 🤝 Support

### Common Commands Summary

```bash
# Complete pipeline
python my-orlm/run_training_pipeline.py

# Resume from specific step
python my-orlm/run_training_pipeline.py --skip-setup --skip-training

# Individual scripts
python my-orlm/setup_environment.py
python my-orlm/train_qlora.py
python my-orlm/merge_model.py
python my-orlm/test_inference.py
python my-orlm/convert_to_gguf.py
python my-orlm/deploy_ollama.py

# Use deployed model
ollama run orlm-qwen3-8b
```

### Getting Help

```bash
# Show help for any script
python my-orlm/run_training_pipeline.py --help
python my-orlm/train_qlora.py --help
python my-orlm/merge_model.py --help
# ... etc
```

---

## 📝 License

This project uses:
- Qwen models (Apache 2.0 / Custom License)
- Check individual component licenses for details

---

## 🎉 You're Ready!

Start with:

```bash
cd ~/work/OR
source venv/bin/activate
python my-orlm/run_training_pipeline.py
```

The automated pipeline will handle everything from setup to deployment. Grab a coffee and let it run! ☕

**Questions?** Check the troubleshooting section or run scripts with `--help` flag.
