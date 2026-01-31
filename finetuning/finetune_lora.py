"""
Fine-tuning FunctionGemma with LoRA for Todo List use case.

This script uses LoRA (Low-Rank Adaptation) to fine-tune FunctionGemma
without catastrophic forgetting. LoRA only trains a small number of
adapter weights while keeping the base model frozen.

Usage:
    python finetune_lora.py

Requirements:
    pip install torch transformers datasets accelerate trl peft tensorboard
"""

import json
import os
from pathlib import Path

# Force GPU 0 (5060 Ti 16GB)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema
from trl import SFTConfig, SFTTrainer

# Verify GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# =============================================================================
# Configuration - Conservative settings to prevent catastrophic forgetting
# =============================================================================

BASE_MODEL = "google/functiongemma-270m-it"
OUTPUT_DIR = "./finetuned-functiongemma-todo-lora"

# Training hyperparameters - MUCH more conservative
LEARNING_RATE = 2e-4  # Higher for LoRA (standard for LoRA)
NUM_EPOCHS = 3        # FEWER epochs to prevent overfitting
BATCH_SIZE = 2        # Smaller batch for better gradient estimates
MAX_LENGTH = 512

# LoRA configuration
LORA_R = 16           # Rank - higher = more capacity
LORA_ALPHA = 32       # Scaling factor
LORA_DROPOUT = 0.05   # Slight dropout for regularization


# =============================================================================
# Tool Definitions (must match our app's tools)
# =============================================================================

def create_task(title: str, due_date: str = None, priority: str = "normal") -> dict:
    """
    Create a new task or add an item to the todo list.
    Use when user says: add, create, new task, remind me, need to, buy, get.
    Examples: add milk, buy groceries, remind me to call mom.

    Args:
        title: The task title or description to add.
        due_date: Optional due date (today, tomorrow, or specific date). Only set if user mentions a date.
        priority: Task priority - low, normal, or high.
    """
    return {"status": "created"}


def list_tasks(status: str = None, priority: str = None) -> dict:
    """
    List and show tasks from the todo list.
    Use when user says: show, list, what do I need, my tasks, what's on my list.
    For urgent/important tasks use priority=high.
    For done/completed tasks use status=completed.

    Args:
        status: Filter by status - pending or completed.
        priority: Filter by priority - low, normal, or high.
    """
    return {"tasks": []}


def complete_task(task_id: int = None, title: str = None) -> dict:
    """
    Mark a task as done or completed.
    Use when user says: finished, done, completed, did, checked off.
    Examples: I finished shopping, mark milk as done, complete task 1.

    Args:
        task_id: The numeric ID of the task to complete.
        title: Part of the task title to match.
    """
    return {"status": "completed"}


def delete_task(task_id: int = None, title: str = None, status: str = None) -> dict:
    """
    Delete or remove tasks from the list.
    Use when user says: delete, remove, clear, erase, get rid of.
    To delete completed tasks use status=completed.

    Args:
        task_id: The numeric ID of the task to delete.
        title: Part of the task title to match.
        status: Delete all tasks with this status (e.g., completed).
    """
    return {"deleted": 0}


def update_task(task_id: int, title: str = None, priority: str = None, due_date: str = None) -> dict:
    """
    Update an existing task's title, priority, or due date.
    Use when user says: update, change, rename, edit, modify, set.

    Args:
        task_id: The numeric ID of the task to update.
        title: New title for the task.
        priority: New priority - low, normal, or high.
        due_date: New due date for the task.
    """
    return {"status": "updated"}


TOOLS = [
    get_json_schema(create_task),
    get_json_schema(list_tasks),
    get_json_schema(complete_task),
    get_json_schema(delete_task),
    get_json_schema(update_task),
]

DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"


# =============================================================================
# Dataset Preparation
# =============================================================================

def load_training_data():
    """Load training data from JSON file."""
    data_path = Path(__file__).parent / "todo_training_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_conversation(sample: dict) -> dict:
    """Convert a training sample to conversation format."""
    return {
        "messages": [
            {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
            {"role": "user", "content": sample["user_content"]},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": sample["tool_name"],
                            "arguments": json.loads(sample["tool_arguments"]),
                        },
                    }
                ],
            },
        ],
        "tools": TOOLS,
    }


def prepare_dataset():
    """Prepare and split the dataset."""
    raw_data = load_training_data()
    print(f"Loaded {len(raw_data)} training examples")

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        create_conversation,
        remove_columns=dataset.features,
        batched=False,
    )

    # 80/20 train/test split
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    print(f"Training samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")

    return dataset


# =============================================================================
# Model Loading with LoRA
# =============================================================================

def load_model_with_lora():
    """Load FunctionGemma model with LoRA adapters."""
    print(f"\nLoading model: {BASE_MODEL}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="auto",
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Base model device: {model.device}")
    print(f"Base model dtype: {model.dtype}")

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers only
        bias="none",
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA applied:")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")

    return model, tokenizer


# =============================================================================
# Training
# =============================================================================

def train(model, tokenizer, dataset):
    """Run the fine-tuning training with LoRA."""
    torch_dtype = model.dtype

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_LENGTH,
        packing=False,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,  # Effective batch size = 4
        gradient_checkpointing=False,
        optim="adamw_torch",  # Standard optimizer
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,  # Warmup for 10% of training
        bf16=True,
        lr_scheduler_type="cosine",  # Cosine decay for smoother training
        report_to="tensorboard",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print("Starting LoRA training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA rank: {LORA_R}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save LoRA adapters and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nLoRA adapters saved to: {OUTPUT_DIR}")

    return trainer


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, tokenizer, dataset, phase=""):
    """Evaluate the model on test set."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating model... {phase}")
    print("=" * 60 + "\n")

    success_count = 0
    total = len(dataset["test"])

    model.eval()

    for idx, item in enumerate(dataset["test"]):
        messages = [
            item["messages"][0],  # developer message
            item["messages"][1],  # user message
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            out = model.generate(
                **inputs.to(model.device),
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,
                do_sample=False,  # Greedy decoding for consistency
            )

        output = tokenizer.decode(
            out[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=False,
        )

        expected_tool = item["messages"][2]["tool_calls"][0]["function"]["name"]
        user_input = item["messages"][1]["content"]

        print(f'{idx + 1}. "{user_input}"')
        print(f"   Expected: {expected_tool}")

        if f"call:{expected_tool}" in output:
            print("   ✅ Correct!")
            success_count += 1
        else:
            # Show a clean snippet of output
            output_clean = output.replace("\n", " ")[:80]
            print(f"   ❌ Wrong - Output: {output_clean}...")

    accuracy = success_count / total * 100
    print(f"\n{'=' * 60}")
    print(f"Results: {success_count}/{total} ({accuracy:.1f}% accuracy)")
    print("=" * 60)

    return accuracy


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 60)
    print("FunctionGemma Fine-tuning with LoRA for Todo List")
    print("=" * 60)

    # Prepare dataset
    dataset = prepare_dataset()

    # Load model with LoRA
    model, tokenizer = load_model_with_lora()

    # Evaluate before training
    print("\n>>> BEFORE FINE-TUNING <<<")
    before_accuracy = evaluate(model, tokenizer, dataset, "(base model)")

    # Train
    train(model, tokenizer, dataset)

    # Evaluate after training
    print("\n>>> AFTER FINE-TUNING <<<")
    after_accuracy = evaluate(model, tokenizer, dataset, "(after LoRA)")

    # Summary
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Before: {before_accuracy:.1f}%")
    print(f"After:  {after_accuracy:.1f}%")
    improvement = after_accuracy - before_accuracy
    print(f"Change: {'+' if improvement >= 0 else ''}{improvement:.1f}%")
    print(f"\nLoRA adapters saved to: {OUTPUT_DIR}")
    print("\nTo merge LoRA into base model for LM Studio:")
    print("  python merge_lora.py")


if __name__ == "__main__":
    main()
