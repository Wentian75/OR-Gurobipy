# 🔄 Training Pipeline Workflow

## Visual Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    STUDENT STARTS HERE                       │
│                                                              │
│         python my-orlm/run_training_pipeline.py              │
│                                                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Environment Setup (5-10 minutes)                   │
│  ┌────────────────────────────────────────────────────┐     │
│  │  setup_environment.py                              │     │
│  │  • Check GPU availability                          │     │
│  │  • Install PyTorch + CUDA                          │     │
│  │  • Install QLoRA dependencies                      │     │
│  │  • Verify installation                             │     │
│  └────────────────────────────────────────────────────┘     │
│  Output: ✓ Ready to train                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: QLoRA Fine-tuning (1-3 hours)                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │  train_qlora.py                                    │     │
│  │  Input:                                            │     │
│  │    • Base model: Qwen/Qwen2.5-7B-Instruct         │     │
│  │    • Data: OR-Instruct-Data-3K-Gurobipy.jsonl     │     │
│  │  Process:                                          │     │
│  │    • Load model in 4-bit quantization             │     │
│  │    • Apply LoRA adapters                          │     │
│  │    • Train on OR problems                         │     │
│  │    • Save adapter weights                         │     │
│  └────────────────────────────────────────────────────┘     │
│  Output: LoRA Adapter (~400MB)                              │
│  Location: ORLM/checkpoints/orlm-qwen3-8b-qlora/           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Merge LoRA Adapter (5-10 minutes)                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │  merge_model.py                                    │     │
│  │  Input:                                            │     │
│  │    • Base model: Qwen/Qwen2.5-7B-Instruct         │     │
│  │    • LoRA adapter (from Step 2)                   │     │
│  │  Process:                                          │     │
│  │    • Load base model                              │     │
│  │    • Load adapter weights                         │     │
│  │    • Merge into single model                      │     │
│  │    • Save full model                              │     │
│  └────────────────────────────────────────────────────┘     │
│  Output: Full Merged Model (~15GB)                          │
│  Location: ORLM/checkpoints/orlm-qwen3-8b-merged/          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Test Inference (2-5 minutes)                       │
│  ┌────────────────────────────────────────────────────┐     │
│  │  test_inference.py                                 │     │
│  │  Input:                                            │     │
│  │    • Merged model (from Step 3)                   │     │
│  │  Process:                                          │     │
│  │    • Load merged model                            │     │
│  │    • Test with sample OR problems                 │     │
│  │    • Generate solutions                           │     │
│  └────────────────────────────────────────────────────┘     │
│  Output: ✓ Model works correctly                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Convert to GGUF (10-20 minutes)                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  convert_to_gguf.py                                │     │
│  │  Input:                                            │     │
│  │    • Merged model (from Step 3)                   │     │
│  │  Process:                                          │     │
│  │    • Clone llama.cpp (if needed)                  │     │
│  │    • Convert to GGUF format (FP16)                │     │
│  │    • Quantize to Q5_K_M (5-bit)                   │     │
│  └────────────────────────────────────────────────────┘     │
│  Output: Quantized GGUF (~5GB)                              │
│  Location: ORLM/checkpoints/gguf/model-Q5_K_M.gguf         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: Deploy to Ollama (2-5 minutes)                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  deploy_ollama.py                                  │     │
│  │  Input:                                            │     │
│  │    • GGUF file (from Step 5)                      │     │
│  │  Process:                                          │     │
│  │    • Create Ollama Modelfile                      │     │
│  │    • Register model with Ollama                   │     │
│  │    • Test deployment                              │     │
│  └────────────────────────────────────────────────────┘     │
│  Output: ✓ Model deployed to Ollama                         │
│  Name: orlm-qwen3-8b                                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    READY TO USE! 🎉                          │
│                                                              │
│              ollama run orlm-qwen3-8b                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## File Size Progression

```
Original Base Model (Download)
    🔽 Qwen/Qwen2.5-7B-Instruct (~15GB, cached by HuggingFace)

Training (Step 2)
    🔽 Creates LoRA Adapter (~400MB)
    📁 ORLM/checkpoints/orlm-qwen3-8b-qlora/

Merging (Step 3)
    🔽 Combines into Full Model (~15GB)
    📁 ORLM/checkpoints/orlm-qwen3-8b-merged/

GGUF Conversion (Step 5)
    🔽 Converts + Quantizes (~5-6GB)
    📁 ORLM/checkpoints/gguf/model-Q5_K_M.gguf

Ollama Deployment (Step 6)
    🔽 Registers with Ollama (no copy, uses GGUF)
    🚀 Ready to use!
```

## Time Breakdown

```
Step 1: Setup           ████░░░░░░░░░░░░░░░░  5-10 min  (one-time)
Step 2: Training        ████████████████████  1-3 hours (main time)
Step 3: Merge           ██░░░░░░░░░░░░░░░░░░  5-10 min
Step 4: Test            █░░░░░░░░░░░░░░░░░░░  2-5 min
Step 5: GGUF Convert    ████░░░░░░░░░░░░░░░░  10-20 min
Step 6: Ollama Deploy   █░░░░░░░░░░░░░░░░░░░  2-5 min
─────────────────────────────────────────────
Total:                  ~2-4 hours (mostly training)
```

## Disk Space Requirements

```
Required Space:
├── Base Model Cache:        ~15 GB  (HuggingFace cache)
├── LoRA Adapter:            ~0.4 GB (Step 2 output)
├── Merged Model:            ~15 GB  (Step 3 output)
├── GGUF Files:              ~5-6 GB (Step 5 output)
└── Working Space:           ~5 GB   (temporary files)
─────────────────────────────────────
Total:                       ~40-45 GB

Optional: After deployment, you can delete:
├── Merged Model (15 GB) - keep GGUF instead
└── Working files (5 GB)
─────────────────────────────────────
Minimum to keep:             ~20-25 GB
```

## GPU Memory Usage

```
Step 2 (Training):
    • Peak VRAM: ~20-24 GB
    • Minimum recommended: 24 GB (RTX 3090, A100, etc.)
    • 4-bit quantization keeps memory low

Step 3 (Merge):
    • Peak VRAM: ~18 GB
    • Uses FP16 precision

Step 4 (Test):
    • Peak VRAM: ~16 GB
    • Inference mode (less memory)

Steps 5-6:
    • Mostly CPU operations
    • Minimal GPU memory
```

## Alternative Paths

### Quick Test Path (Skip Deployment)
```
Step 1 → Step 2 → Step 3 → Step 4
                           STOP HERE
                           (Test only, no Ollama)
```

### Resume from Training
```
Start Here (Skip Step 1 & 2)
     ↓
Step 3 → Step 4 → Step 5 → Step 6
(Use existing adapter)
```

### Custom Parameters Path
```
Step 1 (normal)
     ↓
Step 2 (with custom parameters)
  --epochs 2
  --batch_size 2
  --learning_rate 1e-4
     ↓
Step 3 → 4 → 5 → 6 (normal)
```

## Script Dependencies

```
run_training_pipeline.py (orchestrator)
    │
    ├─→ setup_environment.py
    │       └─→ installs: torch, transformers, peft, etc.
    │
    ├─→ train_qlora.py
    │       ├─→ requires: GPU, training data
    │       └─→ produces: LoRA adapter
    │
    ├─→ merge_model.py
    │       ├─→ requires: base model, adapter
    │       └─→ produces: merged model
    │
    ├─→ test_inference.py
    │       ├─→ requires: merged model
    │       └─→ validates: model quality
    │
    ├─→ convert_to_gguf.py
    │       ├─→ requires: merged model, llama.cpp
    │       └─→ produces: GGUF file
    │
    └─→ deploy_ollama.py
            ├─→ requires: GGUF file, Ollama
            └─→ produces: deployed model
```

## Error Recovery

```
If Training Fails:
    → Check GPU memory
    → Reduce batch_size
    → Resume: python train_qlora.py

If Merge Fails:
    → Check adapter exists
    → Check base model access
    → Resume: python merge_model.py

If GGUF Fails:
    → Check llama.cpp installation
    → Retry conversion
    → Resume: python convert_to_gguf.py

If Ollama Fails:
    → Check Ollama installed
    → Install: curl -fsSL https://ollama.com/install.sh | sh
    → Resume: python deploy_ollama.py
```

## Success Indicators

```
✓ Step 1: "All dependencies installed successfully!"
✓ Step 2: "Training Complete! Model saved to: ..."
✓ Step 3: "Merge Complete! Merged model saved to: ..."
✓ Step 4: "Model loaded successfully"
✓ Step 5: "Conversion Complete! GGUF file: ..."
✓ Step 6: "Deployment Complete! Model name: orlm-qwen3-8b"

Final Test:
$ ollama run orlm-qwen3-8b
✓ Model responds to prompts → SUCCESS! 🎉
```

## Quick Reference Commands

```bash
# Complete pipeline (recommended)
python my-orlm/run_training_pipeline.py

# Resume from specific step
python my-orlm/run_training_pipeline.py --skip-setup --skip-training

# Individual scripts
python my-orlm/setup_environment.py      # Step 1
python my-orlm/train_qlora.py           # Step 2
python my-orlm/merge_model.py           # Step 3
python my-orlm/test_inference.py        # Step 4
python my-orlm/convert_to_gguf.py       # Step 5
python my-orlm/deploy_ollama.py         # Step 6

# Use the model
ollama run orlm-qwen3-8b

# Get help
python my-orlm/run_training_pipeline.py --help
```

---

**Remember**: The automated pipeline handles all of this for you! Just run:
```bash
python my-orlm/run_training_pipeline.py
```

And sit back while it completes all steps automatically! ☕
