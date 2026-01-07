"""
# Product Pricing Prediction with Qwen2.5-VL-7B

[![Weights & Biases](https://img.shields.io/badge/W&B-Tracking-orange?logo=wandb)](https://wandb.ai) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a training script for fine-tuning the Qwen2.5-VL-7B-Instruct model (using Unsloth for efficient 4-bit quantization and LoRA adapters) to predict product market prices in USD. The model processes product listings including images, names, descriptions, bullet points, quantities, and units. It uses a custom MLP regression head for high-precision price prediction and optimizes with Symmetric Mean Absolute Percentage Error (SMAPE) loss, aligning directly with common pricing evaluation metrics.

The script supports image normalization, smart image downloading (only for missing/corrupted files), train/validation splitting, checkpoint resuming, and comprehensive logging via Weights & Biases (W&B) and text files.

## Features
- **Vision-Language Fine-Tuning**: Combines text prompts with product images using Qwen2.5-VL-7B.
- **Efficient Training**: 4-bit quantization, LoRA adapters (r=16), and gradient checkpointing via Unsloth.
- **Custom Regression Head**: MLP for precise price prediction (outputs in float32 with exp() for positivity).
- **Loss Options**: Direct SMAPE or SmoothL1 on log(price) for stability across price ranges.
- **Image Handling**: Automatic download, validation, normalization (resize to 384x384 with padding), and fallback for corrupted images.
- **Dataset Preparation**: Converts CSV data to conversation format; handles missing fields gracefully.
- **Monitoring & Logging**: W&B integration for metrics (losses, gradients, accuracies, price stats), tables, and artifacts; text logging for steps/evals.
- **Checkpointing**: Saves LoRA adapters + MLP head; supports resuming from latest checkpoint.
- **Evaluation**: Periodic validation with SMAPE; tracks best model.

## Requirements
- Python 3.8+
- CUDA-compatible GPU (tested on NVIDIA with 24GB+ VRAM for batch size 2)
- Libraries: See `requirements.txt` (generated from imports):
- torch
transformers
datasets
unsloth
trl
peft
pandas
numpy
pillow
wandb
tqdm

## Installation
1. Clone the repository:
git clone https://github.com/yourusername/product-pricing-qwen.git
cd product-pricing-qwen
2. Install dependencies:pip install -r requirements.txt
3. 
3. Set up Weights & Biases (optional but recommended for logging):
- Sign up at [wandb.ai](https://wandb.ai) and get your API key.
- Run `wandb login` and paste your key.

## Dataset
- **Input Format**: CSV file (e.g., `train_cleaned_final.csv`) with columns:
- `sample_id`: Unique ID.
- `image_link`: URL to product image.
- `value`, `unit`: Extracted quantity (e.g., 500, "ml").
- `bullet_points_list`: List of features (string or literal eval).
- `item_name`: Product title.
- `product_description`: Detailed text.
- `ipq`: Items per quantity (pack size).
- `ipq_missing`: Flag for missing IPQ.
- `price`: Target price in USD (positive float).
- **Images**: Downloaded automatically to `train_images/` folder.
- **Preparation**: Script handles downloading, validation, and conversion to conversation format for training.

## Usage
### Configuration
Edit constants in the script (`new_model.py`):
- `TRAIN_CSV`: Path to training CSV.
- `IMAGE_FOLDER`: Directory for images.
- `OUTPUT_DIR`: Checkpoint output.
- `LORA_MODEL_DIR`: Final model save path.
- W&B: `WANDB_PROJECT`, `WANDB_RUN_NAME`.
- Hyperparameters: `MAX_SAMPLES`, `BATCH_SIZE`, `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, etc.
- `USE_LOG_TARGET`: Toggle between SMAPE (False) or log-space SmoothL1 (True).
- `EVAL_SPLIT`: Validation fraction (default 0.1).

### Training
Run the script:new_model.py

### Inference
Load saved LoRA + MLP and use similar forward pass as in `compute_loss`.

## Acknowledgments
- [Unsloth](https://github.com/unslothai/unsloth): For efficient VL model training.
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct): Base model.
- [Weights & Biases](https://wandb.ai): For experiment tracking.
- Inspired by pricing prediction challenges (e.g., e-commerce competitions).
