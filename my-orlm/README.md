# ORLM Automated Training Scripts

This directory contains automated Python scripts for QLoRA fine-tuning of Qwen3-8B on operations research problems.

## 📦 Available Scripts

### 🌟 Main Entry Point
- **`run_training_pipeline.py`** - Complete automated pipeline orchestrator
  - Runs all steps from setup to deployment
  - Supports resuming from any step
  - Recommended for most users

### 🔧 Individual Components
1. **`setup_environment.py`** - Environment setup and dependency installation
2. **`train_qlora.py`** - QLoRA fine-tuning script
3. **`merge_model.py`** - Merge LoRA adapter with base model
4. **`test_inference.py`** - Test model inference with sample problems
5. **`convert_to_gguf.py`** - Convert model to GGUF format
6. **`deploy_ollama.py`** - Deploy GGUF model to Ollama

### 📄 Configuration Files
- **`requirements.txt`** - Python package dependencies
- **`README.md`** - This file

## 🚀 Quick Start

### Option 1: Automated (Recommended)
```bash
# Run everything automatically
python run_training_pipeline.py
```

### Option 2: Step-by-step
```bash
# 1. Setup (one-time)
python setup_environment.py

# 2. Train
python train_qlora.py

# 3. Merge
python merge_model.py

# 4. Test
python test_inference.py --mode sample

# 5. Convert to GGUF
python convert_to_gguf.py

# 6. Deploy to Ollama
python deploy_ollama.py
```

## 📖 Detailed Documentation

See [QUICK_START.md](../QUICK_START.md) in the parent directory for:
- Complete usage instructions
- Configuration options
- Troubleshooting guide
- Examples and best practices

## 🔍 Script Details

### run_training_pipeline.py
**Purpose**: Orchestrate complete training workflow

**Usage**:
```bash
# Complete pipeline
python run_training_pipeline.py

# Skip already-completed steps
python run_training_pipeline.py --skip-setup --skip-training

# Custom parameters
python run_training_pipeline.py \
    --epochs 2 \
    --batch_size 2 \
    --learning_rate 1e-4
```

**Arguments**:
- `--base_model`: Base model name (default: Qwen/Qwen3-8B)
- `--data_path`: Training data path
- `--batch_size`: Batch size per device
- `--epochs`: Number of training epochs
- `--skip-setup`: Skip environment setup
- `--skip-training`: Skip training step
- `--skip-merge`: Skip model merging
- `--skip-test`: Skip inference testing
- `--skip-gguf`: Skip GGUF conversion
- `--skip-ollama`: Skip Ollama deployment

---

### setup_environment.py
**Purpose**: Install and verify all dependencies

**Usage**:
```bash
python setup_environment.py
```

**What it does**:
- Installs PyTorch with CUDA support
- Installs QLoRA dependencies (transformers, peft, bitsandbytes, etc.)
- Disables W&B logging
- Verifies GPU availability
- Checks installation completeness

---

### train_qlora.py
**Purpose**: QLoRA fine-tuning on OR dataset

**Usage**:
```bash
# Basic training
python train_qlora.py

# Custom parameters
python train_qlora.py \
    --model_name "Qwen/Qwen3-8B" \
    --data_path "ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl" \
    --epochs 2 \
    --batch_size 1 \
    --grad_acc 8 \
    --learning_rate 2e-4 \
    --lora_r 64
```

**Arguments**:
- `--model_name`: Base model to fine-tune
- `--data_path`: Path to training data (JSONL)
- `--output_dir`: Output directory for adapter
- `--batch_size`: Per-device batch size (default: 1)
- `--grad_acc`: Gradient accumulation steps (default: 8)
- `--epochs`: Training epochs (default: 1)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--lora_r`: LoRA rank (default: 64)

**Output**:
- LoRA adapter in `ORLM/checkpoints/orlm-qwen3-8b-qlora/`
- Training logs

---

### merge_model.py
**Purpose**: Merge LoRA adapter with base model

**Usage**:
```bash
# Basic merge
python merge_model.py

# Custom paths
python merge_model.py \
    --base_model "Qwen/Qwen3-8B" \
    --adapter_path "ORLM/checkpoints/orlm-qwen3-8b-qlora" \
    --output_path "ORLM/checkpoints/orlm-qwen3-8b-merged"
```

**Arguments**:
- `--base_model`: Base model name or path
- `--adapter_path`: Path to LoRA adapter
- `--output_path`: Output path for merged model

**Output**:
- Full merged model in `ORLM/checkpoints/orlm-qwen3-8b-merged/`

---

### test_inference.py
**Purpose**: Test model with sample OR problems

**Usage**:
```bash
# Test with sample problems
python test_inference.py --mode sample

# Interactive mode
python test_inference.py --mode interactive

# Both
python test_inference.py --mode both
```

**Arguments**:
- `--model_path`: Path to merged model
- `--mode`: Testing mode (sample/interactive/both)
- `--device`: Device to use (auto/cuda/cpu)

**Modes**:
- `sample`: Test with 3 pre-defined OR problems
- `interactive`: Enter your own problems interactively
- `both`: Sample tests followed by interactive

---

### convert_to_gguf.py
**Purpose**: Convert model to GGUF format for llama.cpp/Ollama

**Usage**:
```bash
# Basic conversion with quantization
python convert_to_gguf.py

# Without quantization
python convert_to_gguf.py --no-quantize

# Custom quantization
python convert_to_gguf.py --quant_type Q4_K_M
```

**Arguments**:
- `--model_path`: Path to merged model
- `--output_dir`: Output directory for GGUF files
- `--llama_cpp_dir`: Path to llama.cpp (auto-cloned if not provided)
- `--quantize`: Quantize the model (default: True)
- `--no-quantize`: Skip quantization
- `--quant_type`: Quantization type (Q4_K_M/Q5_K_M/Q8_0/etc.)

**Output**:
- FP16 GGUF: `ORLM/checkpoints/gguf/model-f16.gguf`
- Quantized: `ORLM/checkpoints/gguf/model-{quant_type}.gguf`

---

### deploy_ollama.py
**Purpose**: Deploy GGUF model to Ollama

**Usage**:
```bash
# Basic deployment
python deploy_ollama.py

# Custom model name
python deploy_ollama.py --model_name my-or-model

# Skip testing
python deploy_ollama.py --no-test
```

**Arguments**:
- `--gguf_path`: Path to GGUF file
- `--model_name`: Name for Ollama model (default: orlm-qwen3-8b)
- `--test`: Test model after deployment (default: True)
- `--no-test`: Skip testing

**Output**:
- Ollama model registered and ready to use
- Modelfile in GGUF directory

**Using deployed model**:
```bash
ollama run orlm-qwen3-8b "Your OR problem here"
```

---

## 💡 Tips

1. **First Time Setup**: Run `setup_environment.py` first
2. **Quick Testing**: Use `--skip-gguf --skip-ollama` to test training faster
3. **Resume Training**: Use `--skip-setup --skip-training` if training already done
4. **Low Memory**: Reduce `--batch_size` or increase `--grad_acc`
5. **Check Progress**: All scripts show detailed progress and status

## 🐛 Common Issues

### Out of Memory
```bash
# Reduce batch size
python train_qlora.py --batch_size 1 --grad_acc 16
```

### CUDA Not Available
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Ollama Not Installed
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

## 📚 Documentation Structure

```
OR/
├── QUICK_START.md          # Main documentation (START HERE!)
├── README.md               # Original manual instructions
└── my-orlm/
    ├── README.md           # This file (script reference)
    ├── requirements.txt    # Python dependencies
    └── *.py               # Automated scripts
```

## 🎯 Workflow Diagram

```
setup_environment.py
        ↓
train_qlora.py (produces LoRA adapter)
        ↓
merge_model.py (merges adapter + base model)
        ↓
test_inference.py (optional testing)
        ↓
convert_to_gguf.py (creates GGUF file)
        ↓
deploy_ollama.py (deploys to Ollama)
        ↓
    ollama run orlm-qwen3-8b
```

## 🔗 Related Files

- **Training Data**: `../ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl`
- **Checkpoints**: `../ORLM/checkpoints/`
- **Documentation**: `../QUICK_START.md`

## 📞 Support

For detailed help on any script:
```bash
python <script_name>.py --help
```

For complete documentation, see [QUICK_START.md](../QUICK_START.md)
