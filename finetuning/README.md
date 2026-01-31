# Fine-tuning FunctionGemma for Todo List

This directory contains scripts and data for fine-tuning FunctionGemma on the todo list use case.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA 12.8+ installed
- Python 3.11+
- HuggingFace account with access to FunctionGemma model

## Installation

### 1. Install PyTorch with CUDA support (IMPORTANT!)

The default pip torch is CPU-only. You must install from PyTorch's CUDA index:

```powershell
# For CUDA 12.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install other fine-tuning dependencies

```powershell
pip install transformers datasets accelerate trl tensorboard protobuf sentencepiece
```

### 3. Login to HuggingFace

You need to accept the FunctionGemma license at: https://huggingface.co/google/functiongemma-270m-it

Then login with a token that has write access:

```powershell
huggingface-cli login
```

## Dataset

The `todo_training_data.json` contains 71 training examples covering:

| Tool | Examples | Sample Phrases |
|------|----------|----------------|
| create_task | 20 | "add milk", "buy eggs", "remind me to..." |
| list_tasks | 16 | "show my tasks", "list urgent tasks" |
| complete_task | 14 | "I finished...", "mark X as done" |
| delete_task | 10 | "delete task 1", "clear completed" |
| update_task | 6 | "change priority", "rename task" |

## Usage

Run the fine-tuning script:

```powershell
cd c:\Users\bukat\Desktop\AI-as-a-Backend
python finetuning/finetune.py
```

Training will:
1. Load the base FunctionGemma 270M model
2. Evaluate accuracy before training
3. Fine-tune for 8 epochs
4. Evaluate accuracy after training
5. Save the model to `./finetuned-functiongemma-todo`

## After Training

### Convert to GGUF for LM Studio

To use the fine-tuned model in LM Studio, you need to convert it to GGUF format:

```powershell
# Clone llama.cpp for conversion tools
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install requirements
pip install -r requirements.txt

# Convert to GGUF
python convert_hf_to_gguf.py ../finetuned-functiongemma-todo --outtype f16

# Optionally quantize to reduce size
./llama-quantize finetuned-functiongemma-todo-f16.gguf finetuned-functiongemma-todo-q4_k_m.gguf q4_k_m
```

### Load in LM Studio

1. Copy the `.gguf` file to LM Studio's models folder
2. Load the model in LM Studio
3. Update `.env` to point to your fine-tuned model:
   ```
   LM_STUDIO_MODEL=finetuned-functiongemma-todo-q4_k_m
   ```

## Expected Results

Based on Google's documentation, fine-tuning should improve accuracy significantly:

| Stage | Expected Accuracy |
|-------|-------------------|
| Before fine-tuning | ~10-30% (varies by phrasing) |
| After fine-tuning | 80-90%+ |

## Troubleshooting

### CUDA not available
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

If False, reinstall PyTorch with CUDA:
```powershell
pip uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Out of Memory
- Reduce `BATCH_SIZE` in finetune.py (try 2 or 1)
- Reduce `MAX_LENGTH` (try 256)
- Enable gradient checkpointing
