# 🚀 ORLM Training Pipeline - Quick Cheat Sheet

## ⚡ TL;DR - Fastest Way to Train

```bash
cd ~/work/OR
source venv/bin/activate
python my-orlm/run_training_pipeline.py
```

**That's it!** Everything runs automatically. ☕ Wait 2-4 hours.

---

## 📋 One-Line Commands

```bash
# Complete automation
python my-orlm/run_training_pipeline.py

# Setup only
python my-orlm/setup_environment.py

# Train only
python my-orlm/train_qlora.py

# Merge only
python my-orlm/merge_model.py

# Test only
python my-orlm/test_inference.py --mode interactive

# GGUF only
python my-orlm/convert_to_gguf.py

# Deploy only
python my-orlm/deploy_ollama.py

# Use model
ollama run orlm-qwen3-8b
```

---

## 🎯 Common Scenarios

### Scenario 1: First Time User
```bash
python my-orlm/run_training_pipeline.py
# Wait for completion, then:
ollama run orlm-qwen3-8b
```

### Scenario 2: Training Already Done
```bash
python my-orlm/run_training_pipeline.py --skip-setup --skip-training
```

### Scenario 3: Just Want to Test
```bash
python my-orlm/test_inference.py --mode interactive
```

### Scenario 4: Custom Training Parameters
```bash
python my-orlm/train_qlora.py \
    --epochs 2 \
    --batch_size 2 \
    --learning_rate 1e-4
```

### Scenario 5: Out of Memory
```bash
python my-orlm/train_qlora.py \
    --batch_size 1 \
    --grad_acc 16
```

### Scenario 6: Quick Validation (No Deployment)
```bash
python my-orlm/run_training_pipeline.py --skip-gguf --skip-ollama
```

---

## 🔧 Key Parameters

### Training Parameters
```bash
--batch_size 1          # Batch size per GPU
--grad_acc 8           # Gradient accumulation (effective batch = batch_size × grad_acc)
--epochs 1             # Training epochs
--learning_rate 2e-4   # Learning rate
--lora_r 64           # LoRA rank (higher = more capacity)
```

### GGUF Quantization Types
```bash
--quant_type Q4_K_M    # Smallest (~4GB)
--quant_type Q5_K_M    # Recommended (~5GB) ⭐
--quant_type Q8_0      # Highest quality (~8GB)
```

### Pipeline Control
```bash
--skip-setup           # Skip environment setup
--skip-training        # Skip training (use existing)
--skip-merge          # Skip merging
--skip-test           # Skip testing
--skip-gguf           # Skip GGUF conversion
--skip-ollama         # Skip Ollama deployment
```

---

## 📁 Key File Locations

```bash
# Training data (you provide this)
ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl

# Outputs (created automatically)
ORLM/checkpoints/orlm-qwen3-8b-qlora/    # LoRA adapter
ORLM/checkpoints/orlm-qwen3-8b-merged/   # Merged model
ORLM/checkpoints/gguf/                    # GGUF files

# Scripts
my-orlm/run_training_pipeline.py         # Main script ⭐
my-orlm/*.py                              # Individual scripts

# Documentation
QUICK_START.md                            # Full guide
my-orlm/README.md                         # Script reference
```

---

## ⏱️ Time & Space Quick Reference

| What | Time | Space |
|------|------|-------|
| Setup | 5-10 min | ~2 GB |
| Training | 1-3 hours | ~15 GB |
| Merge | 5-10 min | ~15 GB |
| Test | 2-5 min | - |
| GGUF | 10-20 min | ~5 GB |
| Deploy | 2-5 min | - |
| **Total** | **~2-4 hours** | **~40 GB** |

---

## 🐛 Quick Troubleshooting

### Problem: Out of Memory
```bash
# Solution 1: Reduce batch size
python my-orlm/train_qlora.py --batch_size 1 --grad_acc 16

# Solution 2: Check GPU
nvidia-smi
```

### Problem: CUDA Not Available
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

### Problem: Training Data Not Found
```bash
# Check file exists
ls -lh ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl

# Verify path in command
python my-orlm/train_qlora.py --data_path "ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl"
```

### Problem: Ollama Not Found
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Or download from
# https://ollama.com/download
```

### Problem: Script Fails Midway
```bash
# Resume from where you left off
python my-orlm/run_training_pipeline.py --skip-setup --skip-training
# (Add --skip-* for completed steps)
```

---

## 💡 Pro Tips

1. **Use the main pipeline**: Let `run_training_pipeline.py` handle everything
2. **Monitor GPU**: Keep `nvidia-smi` running in another terminal
3. **Use screen/tmux**: For long training sessions over SSH
4. **Save space**: Delete merged model after GGUF conversion
5. **Test first**: Use `--skip-gguf --skip-ollama` for quick validation

---

## 🎓 Learning Path

```
Day 1: Setup & Understanding
  → Read QUICK_START.md
  → Run setup_environment.py
  → Verify GPU works

Day 2: Training
  → Start training with train_qlora.py
  → Monitor progress
  → Understand training metrics

Day 3: Deployment
  → Merge and test model
  → Convert to GGUF
  → Deploy to Ollama

Day 4: Usage
  → Use model for OR problems
  → Experiment with prompts
  → Try different parameters
```

---

## 🔍 Validation Checklist

```
Before Training:
[ ] GPU available (nvidia-smi)
[ ] Training data exists
[ ] Python environment activated
[ ] Dependencies installed

After Training:
[ ] LoRA adapter folder exists
[ ] Training log shows loss decreasing
[ ] No error messages in output

After Merge:
[ ] Merged model folder exists (~15GB)
[ ] test_inference.py works

After Deployment:
[ ] ollama list shows your model
[ ] ollama run generates responses
```

---

## 📞 Getting Help

```bash
# Script help
python my-orlm/run_training_pipeline.py --help
python my-orlm/train_qlora.py --help

# Check versions
python --version
python -c "import torch; print(torch.__version__)"
nvidia-smi

# Test imports
python -c "import transformers, peft, bitsandbytes; print('OK')"
```

---

## 🎯 Success Indicators

```
✓ GPU detected
✓ Training loss decreasing
✓ Model generates coherent responses
✓ Ollama model registered
✓ Final test passes

If all ✓ → You're done! 🎉
```

---

## 📊 Resource Requirements

### Minimum
- GPU: 24GB VRAM (RTX 3090, A100)
- RAM: 32GB
- Disk: 50GB free
- Time: 2-4 hours

### Recommended
- GPU: 40GB VRAM (A100)
- RAM: 64GB
- Disk: 100GB free
- Time: Patience ☕

---

## 🚦 Status Check Commands

```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h

# Check training progress
tail -f ORLM/train_qlora.log

# Check if model exists
ls -lh ORLM/checkpoints/

# Check Ollama models
ollama list
```

---

## ⚡ Speed Tips

1. **Use SSD** for faster I/O
2. **Close other GPU apps** to free VRAM
3. **Use batch_size=2** if you have 40GB+ VRAM
4. **Skip testing** in pipeline for faster completion
5. **Use Q4 quantization** instead of Q5 for smaller files

---

## 🎉 You Made It!

**Most important command:**
```bash
python my-orlm/run_training_pipeline.py
```

**Everything else is optional!**

For detailed info, see:
- [QUICK_START.md](QUICK_START.md) - Complete guide
- [my-orlm/README.md](my-orlm/README.md) - Script details
- [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) - Visual guide

---

**Happy Training! 🚀**
