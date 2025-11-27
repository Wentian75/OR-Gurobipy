# Step-by-Step Training Guide

This guide walks you through the complete training process with all the new features.

---

## 📋 Step-by-Step Execution

### Step 1: Setup Environment (One-Time)

```bash
python my-orlm/setup_environment.py
```

**What it does:**
- Installs PyTorch with CUDA support
- Installs QLoRA dependencies (transformers, peft, bitsandbytes, etc.)
- Verifies GPU availability
- Configures environment variables

**Expected output:**
```
✓ All dependencies installed successfully!
✓ GPU: NVIDIA A100-SXM4-40GB
```

---

### Step 2: Train with QLoRA

#### Option A: Interactive Mode (Recommended)

```bash
python my-orlm/train_qlora.py
```

**You will be prompted:**
```
============================================================
Dataset Selection
============================================================

Please select the code style for training:

  1. Gurobi Style  - Uses gurobipy library
     Dataset: OR-Instruct-Data-3K-Gurobipy.jsonl
     Best for: Industry-standard optimization problems

  2. LP Style      - Uses standard LP format
     Dataset: OR-Instruct-Data-3k-LP.jsonl
     Best for: Linear programming problems

Enter your choice (1 or 2):
```

#### Option B: Non-Interactive Mode

```bash
# For Gurobi style
python my-orlm/train_qlora.py --code_style gurobi

# For LP style
python my-orlm/train_qlora.py --code_style lp
```

#### Advanced Training Options

```bash
# Full control with custom parameters
python my-orlm/train_qlora.py \
    --code_style gurobi \
    --epochs 5 \
    --batch_size 1 \
    --grad_acc 8 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --early_stopping \
    --early_stopping_patience 3 \
    --lora_r 64

# Disable early stopping (train for full epochs)
python my-orlm/train_qlora.py \
    --code_style gurobi \
    --no_early_stopping \
    --epochs 3

# Different learning rate schedulers
python my-orlm/train_qlora.py --code_style gurobi --lr_scheduler_type linear
python my-orlm/train_qlora.py --code_style gurobi --lr_scheduler_type cosine_with_restarts
```

**What happens during training:**
1. **Dataset Selection**: Choose Gurobi or LP style (or skip if specified via --code_style)
2. **Data Loading**: Loads 3000 examples
3. **Train/Eval Split**: Automatically splits into 2700 train / 300 eval (90%/10%)
4. **Model Loading**: Loads Qwen3-8B with 4-bit quantization
5. **LoRA Setup**: Applies LoRA adapters (trainable params: ~2.5% of total)
6. **Training**: Trains for up to 5 epochs with cosine learning rate schedule
7. **Early Stopping**: Monitors validation loss, stops if no improvement for 3 epochs
8. **Best Model**: Saves the best checkpoint based on lowest validation loss

**Expected output:**
```
Training Configuration:
============================================================
  Model: Qwen/Qwen3-8B
  Dataset: ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl
  Output: ORLM/checkpoints/orlm-qwen3-8b-qlora
  Batch size: 1
  Gradient accumulation: 8
  Effective batch size: 8
  Epochs: 5
  Learning rate: 0.0002
  LR scheduler: cosine
  Early stopping: Enabled
  Early stopping patience: 3 epochs
  LoRA rank: 64
  LoRA alpha: 128
============================================================

✓ Early stopping enabled with patience=3

🚀 Starting training...

Epoch 1/5:  100%|███████| 338/338 [15:23<00:00]
{'train_loss': 1.234, 'eval_loss': 1.156}

Epoch 2/5:  100%|███████| 338/338 [15:21<00:00]
{'train_loss': 0.987, 'eval_loss': 1.043}  ← improvement!

...
```

**Time estimate:** 1-3 hours (may stop earlier with early stopping)

**Output:** LoRA adapter saved to `ORLM/checkpoints/orlm-qwen3-8b-qlora/`

---

### Step 3: Merge LoRA Adapter with Base Model

```bash
python my-orlm/merge_model.py
```

**Optional parameters:**
```bash
# Custom paths
python my-orlm/merge_model.py \
    --base_model "Qwen/Qwen3-8B" \
    --adapter_path "ORLM/checkpoints/orlm-qwen3-8b-qlora" \
    --output_path "ORLM/checkpoints/orlm-qwen3-8b-merged"
```

**What it does:**
- Loads base model (Qwen3-8B)
- Loads LoRA adapter from Step 2
- Merges adapter weights into base model
- Saves full merged model

**Time estimate:** 5-10 minutes

**Output:** Merged model saved to `ORLM/checkpoints/orlm-qwen3-8b-merged/` (~15GB)

---

### Step 4: Test the Model

#### Option A: Sample Tests

```bash
python my-orlm/test_inference.py --mode sample
```

Tests the model with 3 pre-defined OR problems.

#### Option B: Interactive Testing

```bash
python my-orlm/test_inference.py --mode interactive
```

Enter your own OR problems interactively.

#### Option C: Both

```bash
python my-orlm/test_inference.py --mode both
```

**Example interaction:**
```
Enter your OR problem (or 'quit' to exit):
> A factory produces products A and B. Product A yields $50 profit and requires
2 hours. Product B yields $60 profit and requires 3 hours. There are 100 hours
available. Formulate an optimization problem to maximize profit.

[Model generates solution with mathematical model and Python code]
```

**Time estimate:** 2-5 minutes

---

### Step 5: Convert to GGUF (for Ollama)

```bash
python my-orlm/convert_to_gguf.py
```

**Optional parameters:**
```bash
# Custom quantization type
python my-orlm/convert_to_gguf.py --quant_type Q5_K_M  # Recommended (5-6GB)
python my-orlm/convert_to_gguf.py --quant_type Q4_K_M  # Smaller (4-5GB)
python my-orlm/convert_to_gguf.py --quant_type Q8_0    # Higher quality (8GB)

# Skip quantization (FP16 only)
python my-orlm/convert_to_gguf.py --no-quantize

# Custom paths
python my-orlm/convert_to_gguf.py \
    --model_path "ORLM/checkpoints/orlm-qwen3-8b-merged" \
    --output_dir "ORLM/checkpoints/gguf"
```

**What it does:**
- Clones llama.cpp if needed
- Converts PyTorch model to GGUF format (FP16)
- Quantizes to specified format (Q5_K_M by default)
- Saves GGUF files

**Time estimate:** 10-20 minutes

**Output:**
- FP16 GGUF: `ORLM/checkpoints/gguf/model-f16.gguf`
- Quantized: `ORLM/checkpoints/gguf/model-Q5_K_M.gguf` (~5GB)

---

### Step 6: Deploy to Ollama

```bash
python my-orlm/deploy_ollama.py
```

**Optional parameters:**
```bash
# Custom model name
python my-orlm/deploy_ollama.py --model_name my-or-model

# Specify GGUF file
python my-orlm/deploy_ollama.py \
    --gguf_path "ORLM/checkpoints/gguf/model-Q5_K_M.gguf" \
    --model_name orlm-qwen3-8b

# Skip post-deployment testing
python my-orlm/deploy_ollama.py --no-test
```

**What it does:**
- Checks Ollama installation
- Creates Modelfile with OR-specific system prompt
- Registers model with Ollama
- Tests deployment (optional)

**Time estimate:** 2-5 minutes

**Output:** Model registered with Ollama

---

## 🚀 Using Your Trained Model

### With Ollama

```bash
# Run the model (model name based on dataset choice)
ollama run orlm-qwen3-8b-gurobi  # If you trained with Gurobi style
ollama run orlm-qwen3-8b-lp      # If you trained with LP style

# Example prompt
ollama run orlm-qwen3-8b-gurobi "A company wants to minimize transportation
costs between 3 warehouses and 5 customers. Formulate this as an optimization
problem and provide Python code using gurobipy."

# List all Ollama models
ollama list

# Remove a model
ollama rm orlm-qwen3-8b-gurobi
```

### Direct Inference (Python)

```python
from test_inference import InferenceTester

tester = InferenceTester("ORLM/checkpoints/orlm-qwen3-8b-merged")
tester.load_model()

problem = """
A factory has two production lines. Line 1 can produce 100 units/day at $10/unit.
Line 2 can produce 150 units/day at $8/unit. Daily demand is at least 200 units.
Minimize production costs.
"""

response = tester.generate(problem)
print(response)
```

---

## 📊 Training Progress Example

Here's what you'll see during training with early stopping:

```
Epoch 1/5:
  train_loss: 1.234
  eval_loss: 1.156
  Status: Baseline

Epoch 2/5:
  train_loss: 0.987
  eval_loss: 1.043
  Status: ✓ Improvement! (0.113 decrease)

Epoch 3/5:
  train_loss: 0.845
  eval_loss: 1.052
  Status: ⚠ No improvement (1/3)

Epoch 4/5:
  train_loss: 0.756
  eval_loss: 1.058
  Status: ⚠ No improvement (2/3)

Epoch 5/5:
  train_loss: 0.698
  eval_loss: 1.065
  Status: ⚠ No improvement (3/3)

Early stopping triggered!
Loading best model from Epoch 2...
✓ Training Complete!
```

---

## 🎯 Quick Reference

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--code_style` | Interactive | Dataset choice: `gurobi` or `lp` |
| `--epochs` | 5 | Maximum training epochs |
| `--batch_size` | 1 | Per-device batch size |
| `--grad_acc` | 8 | Gradient accumulation steps |
| `--learning_rate` | 2e-4 | Learning rate |
| `--lr_scheduler_type` | cosine | LR scheduler: cosine, linear, etc. |
| `--early_stopping` | True | Enable early stopping |
| `--early_stopping_patience` | 3 | Epochs without improvement before stopping |
| `--lora_r` | 64 | LoRA rank |
| `--eval_split_ratio` | 0.1 | Validation data ratio (10%) |

### Learning Rate Schedulers

- `linear` - Linear decay from peak to 0
- `cosine` - Cosine annealing (recommended)
- `cosine_with_restarts` - Cosine with periodic restarts
- `polynomial` - Polynomial decay
- `constant` - Constant learning rate

### Quantization Types

- `Q4_K_M` - 4-bit, smallest size (~4GB)
- `Q5_K_M` - 5-bit, good balance (~5GB) ⭐ Recommended
- `Q8_0` - 8-bit, highest quality (~8GB)

---

## 🐛 Troubleshooting

### Issue: "evaluation_strategy" error

**Fixed!** Updated to use `eval_strategy` (newer transformers version).

### Issue: No dataset selection prompt

**Fixed!** Now prompts in Step 2 when running `train_qlora.py` directly.

### Issue: Out of Memory

```bash
# Reduce batch size or increase gradient accumulation
python my-orlm/train_qlora.py --code_style gurobi --batch_size 1 --grad_acc 16
```

### Issue: Early stopping too aggressive

```bash
# Increase patience or disable early stopping
python my-orlm/train_qlora.py --code_style gurobi --early_stopping_patience 5
python my-orlm/train_qlora.py --code_style gurobi --no_early_stopping
```

### Issue: Training too slow

```bash
# Use linear scheduler (faster than cosine)
python my-orlm/train_qlora.py --code_style gurobi --lr_scheduler_type linear
```

---

## 💡 Best Practices

1. **Start with defaults** - They're optimized for most use cases
2. **Monitor early stopping** - It prevents overfitting and saves time
3. **Use cosine scheduler** - Better convergence than linear for long training
4. **Test after merging** - Verify model quality before GGUF conversion
5. **Choose Q5_K_M quantization** - Best balance of size and quality
6. **Save disk space** - Delete merged model after GGUF conversion if needed

---

## 📈 Expected Timeline

| Step | Time | Can Skip? |
|------|------|-----------|
| 1. Setup | 5-10 min | After first time |
| 2. Training | 1-3 hours | No (main step) |
| 3. Merge | 5-10 min | No |
| 4. Test | 2-5 min | Yes (optional) |
| 5. GGUF | 10-20 min | If not using Ollama |
| 6. Deploy | 2-5 min | If not using Ollama |
| **Total** | **~2-4 hours** | |

---

## 🎓 Learning Path

**Day 1**: Steps 1-2 (Setup + Training)
- Understand the dataset selection
- Monitor training progress
- Learn about early stopping behavior

**Day 2**: Steps 3-4 (Merge + Test)
- Test model quality
- Compare outputs for different OR problems
- Iterate on training parameters if needed

**Day 3**: Steps 5-6 (GGUF + Deploy)
- Convert to deployment format
- Deploy to Ollama
- Use the model for real OR problems

---

## ✅ Success Checklist

Before each step, verify:

- [ ] **Step 1**: GPU detected, all packages installed
- [ ] **Step 2**: Dataset selected, training loss decreasing, validation loss monitored
- [ ] **Step 3**: Merged model folder exists (~15GB)
- [ ] **Step 4**: Model generates coherent OR solutions
- [ ] **Step 5**: GGUF file created (~5GB for Q5_K_M)
- [ ] **Step 6**: `ollama list` shows your model

**Final test:**
```bash
ollama run orlm-qwen3-8b-gurobi "test problem"
```

If it responds correctly → **Success!** 🎉

---

For questions or issues, check:
- `QUICK_START.md` - Complete automation guide
- `my-orlm/README.md` - Detailed script reference
- `CLAUDE.md` - Architecture and development guide
