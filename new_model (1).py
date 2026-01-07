# -- coding: utf-8 --
"""
Product Pricing Prediction with Qwen2.5-VL-7B
Training Script with Image Normalization
"""

import os
import sys

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import re
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import ast

# ==================== Configuration ====================
# ==================== Configuration ====================
TRAIN_CSV = "train_cleaned_final.csv"
IMAGE_FOLDER = "train_images"
OUTPUT_DIR = "pricing_model_output"
LORA_MODEL_DIR = "pricing_lora_model"


# W&B Configuration
WANDB_PROJECT = "product-pricing-qwen"  # ← Your project name
WANDB_RUN_NAME = "qwen-7b-40k-samples-3epochs"  # ← Name for this run
WANDB_ENTITY = None  # ← Your wandb username/team (optional)

# Training hyperparameters
MAX_SAMPLES = 40000
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 1.5e-4
MAX_LENGTH = 2048

# Evaluation settings
EVAL_SPLIT = 0.1  # 10% for validation
EVAL_STEPS = 2000 

# Loss function settings
SMAPE_EPSILON = 1e-8  # Small epsilon to avoid division by zero in SMAPE

# Advanced training strategy (from best practices)
# Try both approaches and compare validation SMAPE
USE_LOG_TARGET = False  # If True: train with L1 loss on log(price), may be more stable
                        # If False: train directly with SMAPE loss (current approach)

# MLP Head settings
MLP_INPUT_DIM = 3584  # Qwen2.5-VL-7B hidden size
MLP_HIDDEN_DIMS = [1024, 512, 256]
MLP_DROPOUT = 0.2


# ==================== Helper Functions ====================

def check_image_exists_and_valid(image_link, image_folder):
    """Check if image file exists and is valid (not corrupted/partial)"""
    try:
        filename = Path(image_link).name
        image_path = os.path.join(image_folder, filename)
        
        # Check if file exists
        if not os.path.exists(image_path):
            return False
        
        # Check if file size is reasonable (not empty or too small)
        file_size = os.path.getsize(image_path)
        if file_size < 1024:  # Less than 1KB is suspicious
            print(f"Warning: Image too small, will re-download: {filename}")
            return False
        
        # Try to open image to verify it's valid
        try:
            # Verify image in a way that doesn't corrupt the file handle
            with Image.open(image_path) as img:
                img.verify()
            # Re-open to check it's actually loadable (verify() corrupts handle)
            with Image.open(image_path) as img:
                img.load()  # Force load to detect corruption
            return True
        except Exception as e:
            print(f"Warning: Corrupted image, will re-download: {filename} - {e}")
            return False
            
    except Exception as e:
        print(f"Error checking image: {e}")
        return False

def download_single_image(image_link, image_folder, force_redownload=False):
    """Download a single image if it doesn't exist or is invalid"""
    import urllib.request
    
    try:
        filename = Path(image_link).name
        image_path = os.path.join(image_folder, filename)
        
        # Check if we need to download
        if not force_redownload and check_image_exists_and_valid(image_link, image_folder):
            return True  # Image already exists and is valid
        
        # Download the image
        urllib.request.urlretrieve(image_link, image_path)
        
        # Verify the downloaded image
        if check_image_exists_and_valid(image_link, image_folder):
            return True
        else:
            print(f"Failed to download valid image: {filename}")
            return False
            
    except Exception as e:
        print(f"Error downloading {image_link}: {e}")
        return False

def download_images_smart(df, image_folder, max_workers=20):
    """
    Smart image downloader that:
    - Checks each image individually
    - Only downloads missing/corrupted images
    - Uses cached valid images
    - Shows progress
    """
    import concurrent.futures
    
    # Create folder if needed
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"Created image folder: {image_folder}")
    
    print(f"\nChecking images in {image_folder}...")
    
    # Check which images need downloading
    images_to_download = []
    images_cached = 0
    
    for idx, image_link in enumerate(df['image_link'].tolist()):
        if check_image_exists_and_valid(image_link, image_folder):
            images_cached += 1
        else:
            images_to_download.append(image_link)
    
    print(f"✓ Found {images_cached} cached images")
    print(f"✗ Need to download {len(images_to_download)} images")
    
    if len(images_to_download) == 0:
        print("All images are already downloaded and valid!")
        return
    
    # Download missing images
    print(f"\nDownloading {len(images_to_download)} images...")
    
    downloaded = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_single_image, url, image_folder): url 
            for url in images_to_download
        }
        
        # Process completed downloads with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_url), 
                          total=len(images_to_download)):
            url = future_to_url[future]
            try:
                result = future.result()
                if result:
                    downloaded += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Exception downloading {url}: {e}")
                failed += 1
    
    print(f"\n✓ Successfully downloaded: {downloaded}")
    if failed > 0:
        print(f"✗ Failed to download: {failed}")
    print(f"✓ Total valid images: {images_cached + downloaded}")

def get_image_path(image_link, image_folder):
    """Convert image link to local path"""
    filename = Path(image_link).name
    return os.path.join(image_folder, filename)

def normalize_and_load_image(image_path, target_size=(384, 384)):
    """
    Load and normalize image with proper preprocessing
    - Resize to target size for consistency
    - Convert to RGB
    - Handle corrupted images
    """
    try:
        img = Image.open(image_path)
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize while maintaining aspect ratio
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create a white background and paste the image
        background = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - img.size[0]) // 2, 
                  (target_size[1] - img.size[1]) // 2)
        background.paste(img, offset)
        
        return background
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a blank white image as fallback
        return Image.new('RGB', target_size, (255, 255, 255))

def create_prompt_for_pricing(sample_id, image_link, value, unit,
                              bullet_points_list, item_name, product_description,
                              ipq, ipq_missing):
    """
    Universal prompt that instructs the model to extract pricing-relevant information
    from ALL available fields, since data is often scattered or missing.
    """
    # Parse bullet points safely
    if isinstance(bullet_points_list, str):
        try:
            import ast
            bullet_points_list = ast.literal_eval(bullet_points_list)
        except:
            bullet_points_list = []
    
    # Format fields with clear labels
    value_field = f"{value} {unit}" if (value and unit and unit != 'none') else "not specified in value column"
    ipq_field = f"{ipq} items" if (ipq and ipq > 1) else "single item or not specified"
    
    # Prepare bullet points (limit to avoid token overflow)
    if bullet_points_list and len(bullet_points_list) > 0:
        bullet_text = "\n".join(f"  • {point}" for point in bullet_points_list[:8])
    else:
        bullet_text = "  • (no bullet points provided)"
    
    # Truncate description if needed
    desc_text = str(product_description)[:700] if product_description else "(no description provided)"
    
    prompt = f"""Analyze this product listing and predict its market price in USD.

IMPORTANT: Product specifications (quantity, size, pack count) may appear in ANY of the fields below. Look across ALL fields to understand the complete product offering.

PRODUCT NAME:
{item_name}

EXTRACTED QUANTITY (may be incomplete):
• Value/Unit: {value_field}
• Pack Size: {ipq_field}

PRODUCT FEATURES:
{bullet_text}

PRODUCT DESCRIPTION:
{desc_text}

PRICING INSTRUCTIONS:
1. First, identify the ACTUAL product quantity by checking:
   - The item name (look for "oz", "lb", "ml", "count", "pack of X" or other unit)
   - The bullet points (look for size, quantity, or pack information)
   - The description (look for size specifications)

2. Determine the product category from the image and text:
   - Food/beverage items
   - Health/beauty products  
   - Household goods
   - Other categories

3. Assess brand positioning:
   - Premium/specialty brands command higher prices
   - Store/generic brands are more affordable
   - Look for quality indicators in the description

4. Consider pack economics:
   - Multi-packs typically offer volume discounts
   - Single specialty items are usually priced higher per unit

5. Cross-reference the image:
   - Packaging quality indicates price tier
   - Product category confirmation from visual

Based on ALL available information, predict the market price:

Price (USD):"""
    
    return prompt

def convert_to_conversation(sample, image_folder):
    """Convert sample to conversation format for training"""
    image_path = get_image_path(sample['image_link'], image_folder)
    
    # Check if image exists before loading
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load and normalize image
    image = normalize_and_load_image(image_path)
    
    # Extract all features from sample with safe defaults
    sample_id = sample.get('sample_id', 'unknown')
    image_link = sample.get('image_link', '')
    value = sample.get('value', -1)
    unit = sample.get('unit', 'unknown')
    
    # Parse bullet_points_list (might be string representation of list)
    bullet_points_list = sample.get('bullet_points_list', [])
    if isinstance(bullet_points_list, str):
        try:
            bullet_points_list = ast.literal_eval(bullet_points_list)
        except:
            bullet_points_list = []
    
    item_name = sample.get('item_name', '')
    product_description = sample.get('product_description', '')
    ipq = sample.get('ipq', -1)
    ipq_missing = sample.get('ipq_missing', 1)
    
    # Create prompt with all features
    prompt = create_prompt_for_pricing(
        sample_id=sample_id,
        image_link=image_link,
        value=value,
        unit=unit,
        bullet_points_list=bullet_points_list,
        item_name=item_name,
        product_description=product_description,
        ipq=ipq,
        ipq_missing=ipq_missing
    )
    
    # Format price with full precision (15 decimal places, no dollar sign)
    price_str = f"{sample['price']:.15f}"
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": price_str}
            ]
        },
    ]
    
    return {"messages": conversation}

def prepare_dataset(csv_path, image_folder, max_samples=None):
    """Load and prepare dataset with smart image handling"""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing critical values
    # Note: We only require image_link and price; other fields have defaults
    df = df.dropna(subset=['image_link', 'price'])
    
    # Filter out invalid prices
    df = df[df['price'] > 0]
    
    if max_samples:
        df = df.head(max_samples)
    
    print(f"Dataset size: {len(df)} samples")
    
    # Smart image download - only downloads missing/corrupted images
    download_images_smart(df, image_folder)
    
    # Verify all images are available before processing
    print("\nVerifying all images are available...")
    missing_images = []
    for idx, row in df.iterrows():
        image_path = get_image_path(row['image_link'], image_folder)
        if not check_image_exists_and_valid(row['image_link'], image_folder):
            missing_images.append(idx)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images are still missing or invalid")
        print(f"Removing samples with missing images...")
        df = df.drop(missing_images)
        print(f"New dataset size: {len(df)} samples")
    
    # Convert to conversation format
    print("\nConverting to conversation format...")
    converted_dataset = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        try:
            conv = convert_to_conversation(row, image_folder)
            converted_dataset.append(conv)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"\n✓ Successfully converted {len(converted_dataset)} samples")
    return converted_dataset



def prepare_dataset_with_split(csv_path, image_folder, max_samples=None, eval_split=0.1):
    """Load and prepare dataset with train/validation split"""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing critical values
    # Note: We only require image_link and price; other fields have defaults
    df = df.dropna(subset=['image_link', 'price'])
    
    # Filter out invalid prices
    df = df[df['price'] > 0]
    
    if max_samples:
        df = df.head(max_samples)
    
    print(f"Dataset size: {len(df)} samples")
    
    # Smart image download - only downloads missing/corrupted images
    download_images_smart(df, image_folder)
    
    # Verify all images are available before processing
    print("\nVerifying all images are available...")
    missing_images = []
    for idx, row in df.iterrows():
        image_path = get_image_path(row['image_link'], image_folder)
        if not check_image_exists_and_valid(row['image_link'], image_folder):
            missing_images.append(idx)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images are still missing or invalid")
        print(f"Removing samples with missing images...")
        df = df.drop(missing_images)
        print(f"New dataset size: {len(df)} samples")
    
    # Split into train and validation
    print(f"\nSplitting dataset (train: {1-eval_split:.1%}, val: {eval_split:.1%})...")
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=3407).reset_index(drop=True)
    split_idx = int(len(df) * (1 - eval_split))
    
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Convert to conversation format
    def convert_df_to_dataset(dataframe, desc):
        converted_dataset = []
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc):
            try:
                conv = convert_to_conversation(row, image_folder)
                converted_dataset.append(conv)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        return converted_dataset
    
    print("\nConverting train set to conversation format...")
    train_dataset = convert_df_to_dataset(train_df, "Processing train samples")
    
    print("\nConverting validation set to conversation format...")
    val_dataset = convert_df_to_dataset(val_df, "Processing val samples")
    
    print(f"\n✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


class PriceRegressionHead(nn.Module):
    """
    MLP head for high-precision price prediction
    Takes embeddings from Qwen2.5-VL and outputs float32 price
    """
    def __init__(
        self, 
        input_dim=3584,
        hidden_dims=[1024, 512, 256],
        dropout=0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: Embeddings [batch_size, hidden_dim]
        Returns:
            prices: [batch_size] with float32 precision (15 decimal places)
        
        Note: Uses exp() to ensure positive prices without saturation issues.
        The MLP learns to output log(price), then exp() converts to actual price.
        This approach:
        - Ensures positive prices (exp always > 0)
        - No gradient saturation (unlike softplus)
        - Stable training across wide price ranges ($0.50 to $500+)
        """
        log_price = self.mlp(x).squeeze(-1)
        
        # Convert log-space to price-space using exp()
        # This naturally handles positive prices without saturation
        price = torch.exp(log_price)
        
        return price


class EvalLossCallback(TrainerCallback):
    """Custom callback to track and display training and eval losses"""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens"""
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
            
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)
                print(f"\n{'='*60}")
                print(f"📊 Evaluation at Step {state.global_step}")
                print(f"{'='*60}")
                print(f"Eval Loss: {logs['eval_loss']:.4f}")
                if len(self.train_losses) > 0:
                    print(f"Latest Train Loss: {self.train_losses[-1]:.4f}")
                print(f"{'='*60}\n")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics is not None and 'eval_loss' in metrics:
            print(f"\n✓ Evaluation completed - Loss: {metrics['eval_loss']:.4f}\n")



class TextFileLoggingCallback(TrainerCallback):
    """Callback to log training progress to a text file"""
    
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file
        # Initialize log file
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Training Session Started: {pd.Timestamp.now()}\n")
            f.write(f"{'='*80}\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to text file"""
        if logs is not None:
            with open(self.log_file, 'a') as f:
                timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] Step {state.global_step} | ")
                
                # Log available metrics
                if 'loss' in logs:
                    f.write(f"Loss: {logs['loss']:.4f} | ")
                if 'learning_rate' in logs:
                    f.write(f"LR: {logs['learning_rate']:.2e} | ")
                if 'epoch' in logs:
                    f.write(f"Epoch: {logs['epoch']:.2f} | ")
                
                f.write("\n")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics"""
        if metrics is not None:
            with open(self.log_file, 'a') as f:
                timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"\n[{timestamp}] EVALUATION at Step {state.global_step}\n")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log training completion"""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Training Completed: {pd.Timestamp.now()}\n")
            f.write(f"Total Steps: {state.global_step}\n")
            f.write(f"{'='*80}\n\n")


class MLPCheckpointCallback(TrainerCallback):
    """Callback to save MLP head with each checkpoint"""
    
    def __init__(self, mlp_head, mlp_config):
        self.mlp_head = mlp_head
        self.mlp_config = mlp_config
    
    def on_save(self, args, state, control, **kwargs):
        """Save MLP head whenever a checkpoint is saved"""
        # Get checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save MLP head
        mlp_path = os.path.join(checkpoint_dir, "mlp_head.pt")
        torch.save({
            'state_dict': self.mlp_head.state_dict(),
            'config': self.mlp_config,
        }, mlp_path)
        
        print(f"✓ Saved MLP head to {mlp_path}")


class WandbEvalCallback(TrainerCallback):
    """Enhanced callback for W&B logging with custom metrics"""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []
        self.best_eval_loss = float('inf')
        self.eval_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens"""
        if logs is not None:
            # Track losses locally
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.steps.append(state.global_step)
                
                # Track throughput
                import time
                current_time = time.time()
                elapsed_since_start = current_time - self.train_start_time
                elapsed_since_log = current_time - self.last_log_time
                self.samples_processed += BATCH_SIZE * GRADIENT_ACCUMULATION
                
                # Calculate throughput
                samples_per_sec = self.samples_processed / elapsed_since_start if elapsed_since_start > 0 else 0
                
                # Estimate remaining time
                if state.max_steps and state.max_steps > 0:
                    steps_remaining = state.max_steps - state.global_step
                    steps_per_sec = state.global_step / elapsed_since_start if elapsed_since_start > 0 else 0
                    hours_remaining = (steps_remaining / steps_per_sec) / 3600 if steps_per_sec > 0 else 0
                    
                    # Log throughput metrics
                    wandb.log({
                        "perf/samples_per_second": samples_per_sec,
                        "perf/steps_per_second": steps_per_sec,
                        "perf/hours_elapsed": elapsed_since_start / 3600,
                        "perf/hours_remaining_estimate": hours_remaining,
                        "perf/progress_percent": (state.global_step / state.max_steps) * 100,
                    }, step=state.global_step)
                
                self.last_log_time = current_time
            
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)
                
                # Check if this is the best model
                is_best = logs['eval_loss'] < self.best_eval_loss
                if is_best:
                    self.best_eval_loss = logs['eval_loss']
                    wandb.run.summary["best_eval_loss"] = self.best_eval_loss
                    wandb.run.summary["best_eval_step"] = state.global_step
                
                # Calculate improvement metrics
                improvement = None
                if len(self.eval_losses) > 1:
                    prev_loss = self.eval_losses[-2]
                    improvement = ((prev_loss - logs['eval_loss']) / prev_loss) * 100
                
                # Log to W&B with enhanced metrics
                eval_log_dict = {
                    "eval/loss": logs['eval_loss'],
                    "eval/best_loss": self.best_eval_loss,
                    "eval/is_best_model": 1.0 if is_best else 0.0,
                }
                
                if improvement is not None:
                    eval_log_dict["eval/improvement_percent"] = improvement
                
                # Add training progress
                if len(self.train_losses) > 0:
                    eval_log_dict["eval/train_eval_gap"] = abs(self.train_losses[-1] - logs['eval_loss'])
                    eval_log_dict["eval/overfitting_ratio"] = logs['eval_loss'] / (self.train_losses[-1] + 1e-8)
                
                wandb.log(eval_log_dict, step=state.global_step)
                
                # Store eval history
                self.eval_history.append({
                    'step': state.global_step,
                    'loss': logs['eval_loss'],
                    'is_best': is_best
                })
                
                # Print to console with enhanced info
                print(f"\n{'='*60}")
                print(f"📊 Evaluation at Step {state.global_step}")
                print(f"{'='*60}")
                print(f"Eval Loss:      {logs['eval_loss']:.4f} {'🏆 NEW BEST!' if is_best else ''}")
                print(f"Best Eval Loss: {self.best_eval_loss:.4f}")
                if improvement is not None:
                    emoji = "📈" if improvement > 0 else "📉"
                    print(f"Improvement:    {improvement:+.2f}% {emoji}")
                if len(self.train_losses) > 0:
                    print(f"Latest Train Loss: {self.train_losses[-1]:.4f}")
                    print(f"Train/Eval Gap:    {abs(self.train_losses[-1] - logs['eval_loss']):.4f}")
                print(f"{'='*60}\n")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Log training configuration at start"""
        import time
        self.train_start_time = time.time()
        self.last_log_time = time.time()
        self.samples_processed = 0
        
        wandb.config.update({
            "max_samples": MAX_SAMPLES,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "num_epochs": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "eval_split": EVAL_SPLIT,
            "eval_steps": EVAL_STEPS,
        })
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log comprehensive final summary"""
        import time
        total_time = time.time() - self.train_start_time
        
        summary_dict = {
            "training/total_time_hours": total_time / 3600,
            "training/total_steps": state.global_step,
        }
        
        if len(self.eval_losses) > 0:
            summary_dict["final_eval_loss"] = self.eval_losses[-1]
            summary_dict["best_eval_loss"] = self.best_eval_loss
            summary_dict["total_eval_runs"] = len(self.eval_losses)
            summary_dict["eval_improvement_total"] = ((self.eval_losses[0] - self.eval_losses[-1]) / self.eval_losses[0]) * 100
            
            wandb.run.summary.update(summary_dict)
            
        if len(self.train_losses) > 0:
            wandb.run.summary["final_train_loss"] = self.train_losses[-1]
            wandb.run.summary["best_train_loss"] = min(self.train_losses)
            wandb.run.summary["train_improvement_total"] = ((self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0]) * 100
        
        # Log evaluation history as a table
        if len(self.eval_history) > 0:
            eval_table = wandb.Table(
                data=[[e['step'], e['loss'], e['is_best']] for e in self.eval_history],
                columns=["step", "eval_loss", "is_best"]
            )
            wandb.log({"eval/history_table": eval_table})


def extract_price_from_text(text):
    """
    Extract numeric price from text output
    Examples: "10.990000000000000" -> 10.99, "25.500000000000000" -> 25.50
    Handles both full precision (15 decimals) and standard formats
    """
    try:
        # Remove dollar signs (if any) and commas, then extract numbers
        text = str(text).replace('$', '').replace(',', '').strip()
        # Find all numeric patterns (including decimals)
        matches = re.findall(r'\d+\.?\d*', text)
        if matches:
            return float(matches[0])
        return None
    except:
        return None


def smape_loss(predictions, targets, epsilon=1e-8):
    """
    Compute SMAPE (Symmetric Mean Absolute Percentage Error) loss.
    
    Formula (per competition best practices):
        SMAPE = 2 * |y_true - y_pred| / (|y_true| + |y_pred| + ε)
    
    Args:
        predictions: Predicted prices (tensor)
        targets: Actual prices (tensor)
        epsilon: Small value to avoid division by zero (added inside denominator)
    
    Returns:
        SMAPE loss value (scalar tensor)
    
    Note: 
    - PyTorch autograd handles absolute-value subgradients automatically
    - Epsilon is placed inside the sum for numerical stability
    - This directly matches the competition evaluation metric
    """
    # Compute numerator: 2 * |y_true - y_pred|
    numerator = 2.0 * torch.abs(predictions - targets)
    
    # Compute denominator: |y_true| + |y_pred| + ε
    # Epsilon is added INSIDE the sum for proper numerical stability
    denominator = torch.abs(targets) + torch.abs(predictions) + epsilon
    
    # Compute SMAPE per sample
    smape = numerator / denominator
    
    # Return mean SMAPE across batch
    return torch.mean(smape)


class CustomPricingTrainer(SFTTrainer):
    """
    Custom trainer that uses MLP head for direct price regression with SMAPE loss only.
    
    Architecture:
    - Qwen2.5-VL (4-bit quantized + LoRA) → Embeddings
    - MLP Head → Price prediction
    - Loss: SMAPE only (no cross-entropy)
    
    SMAPE directly aligns with the competition evaluation metric!
    """
    
    def __init__(self, *args, mlp_head=None, **kwargs):
        self.smape_epsilon = kwargs.pop('smape_epsilon', SMAPE_EPSILON)
        super().__init__(*args, **kwargs)
        
        # Attach MLP head
        self.mlp_head = mlp_head
        if self.mlp_head is None:
            raise ValueError("mlp_head is required!")
        
        # Ensure MLP is on correct device and trainable
        self.mlp_head.to(self.model.device)
        self.mlp_head.train()
        
        # Track losses
        self.smape_losses = []
        self.epoch_losses = []
    
    def create_optimizer(self):
        """
        Create custom optimizer that includes both LoRA and MLP parameters.
        MLP gets higher learning rate since it's training from scratch.
        Uses AdamW optimizer for both parameter groups.
        """
        # Separate parameter groups for different learning rates
        optimizer_grouped_parameters = [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": self.mlp_head.parameters(),
                "lr": self.args.learning_rate * 2.0,  # 2x higher LR for MLP (trains from scratch)
                "weight_decay": self.args.weight_decay * 0.5,  # Lower weight decay for MLP
            }
        ]
        
        # Create AdamW optimizer with custom parameter groups
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Print optimizer configuration
        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        mlp_params = sum(p.numel() for p in self.mlp_head.parameters())
        
        print(f"\n{'='*60}")
        print("✓ Custom Optimizer Created (AdamW)")
        print(f"{'='*60}")
        print(f"LoRA Parameters:    {lora_params:,} ({lora_params/1e6:.2f}M)")
        print(f"  → Learning Rate:  {self.args.learning_rate:.2e}")
        print(f"  → Weight Decay:   {self.args.weight_decay}")
        print(f"")
        print(f"MLP Parameters:     {mlp_params:,} ({mlp_params/1e6:.2f}M)")
        print(f"  → Learning Rate:  {self.args.learning_rate * 2:.2e} (2x higher)")
        print(f"  → Weight Decay:   {self.args.weight_decay * 0.5} (0.5x lower)")
        print(f"")
        print(f"Total Trainable:    {lora_params + mlp_params:,} ({(lora_params + mlp_params)/1e6:.2f}M)")
        print(f"Loss Function:      SMAPE Only (direct regression)")
        print(f"{'='*60}\n")
        
        return optimizer
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Custom evaluation loop that properly sets MLP to eval mode
        """
        # Set MLP to eval mode during evaluation
        self.mlp_head.eval()
        try:
            # Call parent evaluation loop
            output = super().evaluation_loop(
                dataloader, 
                description, 
                prediction_loss_only, 
                ignore_keys, 
                metric_key_prefix
            )
            return output
        finally:
            # Always set MLP back to train mode after evaluation
            self.mlp_head.train()
    
    def training_step(self, model, inputs):
        """
        Override training step to add gradient and learning rate tracking
        """
        # Perform standard training step
        loss = super().training_step(model, inputs)
        
        # Log gradients and learning rates every 50 steps
        if self.state.global_step % 50 == 0 and self.state.global_step > 0:
            # Calculate gradient norms
            lora_grad_norm = 0.0
            mlp_grad_norm = 0.0
            
            # LoRA gradients
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    lora_grad_norm += p.grad.norm(2).item() ** 2
            
            # MLP gradients
            for p in self.mlp_head.parameters():
                if p.grad is not None:
                    mlp_grad_norm += p.grad.norm(2).item() ** 2
            
            lora_grad_norm = lora_grad_norm ** 0.5
            mlp_grad_norm = mlp_grad_norm ** 0.5
            
            # Get current learning rates from optimizer
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                lora_lr = self.optimizer.param_groups[0]['lr']
                mlp_lr = self.optimizer.param_groups[1]['lr']
            else:
                lora_lr = self.args.learning_rate
                mlp_lr = self.args.learning_rate * 2.0
            
            # Log to W&B
            wandb.log({
                "gradients/lora_norm": lora_grad_norm,
                "gradients/mlp_norm": mlp_grad_norm,
                "gradients/ratio_mlp_to_lora": mlp_grad_norm / (lora_grad_norm + 1e-8),
                "gradients/total_norm": (lora_grad_norm**2 + mlp_grad_norm**2)**0.5,
                "training/lora_lr": lora_lr,
                "training/mlp_lr": mlp_lr,
                "training/lr_ratio": mlp_lr / lora_lr,
            }, step=self.state.global_step)
        
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Forward pass through model + MLP, compute SMAPE loss only
        """
        try:
            # 1. Forward pass through Qwen2.5-VL (4-bit quantized + LoRA)
            outputs = model(
                input_ids=inputs.get('input_ids'),
                attention_mask=inputs.get('attention_mask'),
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                output_hidden_states=True,  # Need hidden states for MLP
            )
            
            # 2. Extract embeddings from last layer
            # Shape: [batch_size, seq_length, hidden_dim=3584]
            hidden_states = outputs.hidden_states[-1]
            
            # 3. Pool embeddings (mean pooling over sequence)
            # Shape: [batch_size, 3584]
            # Note: activations are float16/float32, not quantized!
            pooled_embeddings = hidden_states.mean(dim=1)
            
            # 4. Pass through MLP head to get price predictions
            # Shape: [batch_size]
            # MLP operates in float32 for precision
            predicted_prices = self.mlp_head(pooled_embeddings.float())
            
            # 5. Extract true prices from labels and track valid indices
            processor = getattr(self, 'processing_class', None) or self.tokenizer
            labels = inputs['labels']
            
            true_prices = []
            valid_indices = []
            batch_size = labels.shape[0]
            
            for i in range(batch_size):
                # Decode label tokens to text
                label_tokens = labels[i][labels[i] != -100]
                label_text = processor.decode(label_tokens, skip_special_tokens=True)
                
                # Extract numeric price
                true_price = extract_price_from_text(label_text)
                if true_price is not None and true_price > 0:
                    true_prices.append(true_price)
                    valid_indices.append(i)
            
            if len(true_prices) == 0:
                # No valid prices in batch, return dummy loss
                print("⚠️  WARNING: No valid prices found in batch")
                return torch.tensor(0.0, requires_grad=True, device=model.device)
            
            # Validate extraction rate
            extraction_rate = len(true_prices) / batch_size
            if extraction_rate < 0.5:
                print(f"⚠️  WARNING: Low price extraction rate: {len(true_prices)}/{batch_size} ({extraction_rate:.1%})")
            
            # Log extraction metrics occasionally
            if self.state.global_step % 100 == 0:
                wandb.log({
                    "data/price_extraction_rate": extraction_rate,
                    "data/valid_prices_per_batch": len(true_prices),
                }, step=self.state.global_step)
            # 6. Convert to tensors and compute SMAPE loss
            # Only use predictions for samples with valid price labels
            valid_indices_tensor = torch.tensor(valid_indices, device=predicted_prices.device, dtype=torch.long)
            predicted_prices = predicted_prices[valid_indices_tensor]
            
            true_prices_tensor = torch.tensor(
                true_prices, 
                device=predicted_prices.device,
                dtype=torch.float32
            )
            
            # 7. Compute loss based on training strategy
            if USE_LOG_TARGET:
                # Advanced approach: Train with SmoothL1 loss in log-space
                # This can be more stable for wide price ranges
                log_predictions = torch.log(predicted_prices + 1e-8)
                log_targets = torch.log(true_prices_tensor + 1e-8)
                
                # SmoothL1 (Huber) loss in log space
                loss = F.smooth_l1_loss(log_predictions, log_targets, beta=1.0)
                
                # Compute SMAPE for monitoring (not for training)
                smape_metric = smape_loss(predicted_prices, true_prices_tensor, epsilon=self.smape_epsilon)
            else:
                # Standard approach: Direct SMAPE loss
                loss = smape_loss(
                    predicted_prices, 
                    true_prices_tensor, 
                    epsilon=self.smape_epsilon
                )
                smape_metric = loss  # Same as training loss
            
            # 8. Log comprehensive metrics
            if self.state.global_step % 10 == 0:
                self.smape_losses.append(loss.item())
                
                # Calculate existing metrics
                mae = torch.abs(predicted_prices - true_prices_tensor).mean().item()
                mape = (torch.abs(predicted_prices - true_prices_tensor) / (true_prices_tensor + 1e-8)).mean().item() * 100
                
                # NEW: Relative errors for better pricing analysis
                relative_errors = torch.abs(predicted_prices - true_prices_tensor) / (true_prices_tensor + 1e-8)
                
                # NEW: Price range analysis (low: <$10, mid: $10-$50, high: >$50)
                low_mask = true_prices_tensor < 10
                mid_mask = (true_prices_tensor >= 10) & (true_prices_tensor < 50)
                high_mask = true_prices_tensor >= 50
                
                # Build comprehensive log dictionary
                log_dict = {
                    # Core loss metrics
                    "loss/training": loss.item(),
                    "loss/smape_metric": smape_metric.item(),
                    
                    # Prediction statistics
                    "predictions/mean": predicted_prices.mean().item(),
                    "predictions/std": predicted_prices.std().item(),
                    "predictions/min": predicted_prices.min().item(),
                    "predictions/max": predicted_prices.max().item(),
                    "predictions/median": torch.median(predicted_prices).item(),
                    
                    # Prediction percentiles
                    "predictions/p25": torch.quantile(predicted_prices, 0.25).item(),
                    "predictions/p75": torch.quantile(predicted_prices, 0.75).item(),
                    "predictions/p90": torch.quantile(predicted_prices, 0.90).item(),
                    "predictions/p99": torch.quantile(predicted_prices, 0.99).item(),
                    
                    # Target statistics
                    "targets/mean": true_prices_tensor.mean().item(),
                    "targets/std": true_prices_tensor.std().item(),
                    "targets/min": true_prices_tensor.min().item(),
                    "targets/max": true_prices_tensor.max().item(),
                    "targets/median": torch.median(true_prices_tensor).item(),
                    
                    # Error metrics
                    "error/mae": mae,
                    "error/mape": mape,
                    "error/relative_mean": relative_errors.mean().item(),
                    "error/relative_median": torch.median(relative_errors).item(),
                    "error/relative_p95": torch.quantile(relative_errors, 0.95).item(),
                    "error/relative_max": relative_errors.max().item(),
                    
                    # Accuracy metrics (% of predictions within X% of truth)
                    "accuracy/within_5_percent": (relative_errors < 0.05).float().mean().item() * 100,
                    "accuracy/within_10_percent": (relative_errors < 0.10).float().mean().item() * 100,
                    "accuracy/within_20_percent": (relative_errors < 0.20).float().mean().item() * 100,
                    "accuracy/within_50_percent": (relative_errors < 0.50).float().mean().item() * 100,
                    
                    # Price range distribution
                    "data/low_price_count": low_mask.sum().item(),
                    "data/mid_price_count": mid_mask.sum().item(),
                    "data/high_price_count": high_mask.sum().item(),
                    "data/batch_price_range": (true_prices_tensor.max() - true_prices_tensor.min()).item(),
                }
                
                # Add price-range-specific SMAPE
                if low_mask.sum() > 0:
                    log_dict["error/smape_low_prices"] = smape_loss(
                        predicted_prices[low_mask], true_prices_tensor[low_mask], epsilon=self.smape_epsilon
                    ).item()
                    log_dict["predictions/mean_low_prices"] = predicted_prices[low_mask].mean().item()
                    log_dict["targets/mean_low_prices"] = true_prices_tensor[low_mask].mean().item()
                
                if mid_mask.sum() > 0:
                    log_dict["error/smape_mid_prices"] = smape_loss(
                        predicted_prices[mid_mask], true_prices_tensor[mid_mask], epsilon=self.smape_epsilon
                    ).item()
                    log_dict["predictions/mean_mid_prices"] = predicted_prices[mid_mask].mean().item()
                    log_dict["targets/mean_mid_prices"] = true_prices_tensor[mid_mask].mean().item()
                
                if high_mask.sum() > 0:
                    log_dict["error/smape_high_prices"] = smape_loss(
                        predicted_prices[high_mask], true_prices_tensor[high_mask], epsilon=self.smape_epsilon
                    ).item()
                    log_dict["predictions/mean_high_prices"] = predicted_prices[high_mask].mean().item()
                    log_dict["targets/mean_high_prices"] = true_prices_tensor[high_mask].mean().item()
                
                # MLP-specific metrics
                with torch.no_grad():
                    mlp_input_norm = pooled_embeddings.float().norm(dim=1).mean().item()
                    log_price_values = torch.log(predicted_prices + 1e-8)
                    
                log_dict["mlp/input_norm"] = mlp_input_norm
                log_dict["mlp/output_mean"] = predicted_prices.mean().item()
                log_dict["mlp/output_std"] = predicted_prices.std().item()
                log_dict["mlp/log_price_mean"] = log_price_values.mean().item()
                log_dict["mlp/log_price_std"] = log_price_values.std().item()
                
                # Handle log-target case
                if USE_LOG_TARGET:
                    log_dict["loss/smoothl1_log"] = loss.item()
                    log_dict["loss/smape_monitoring"] = smape_metric.item()
                
                wandb.log(log_dict, step=self.state.global_step)
            
            # Periodic detailed logging (every 100 steps) - scatter plot data
            if self.state.global_step % 100 == 0 and self.state.global_step > 0:
                # Create comparison table for visualization
                comparison_data = []
                num_samples = min(50, len(predicted_prices))  # Log first 50 samples
                for i in range(num_samples):
                    comparison_data.append([
                        true_prices_tensor[i].item(),
                        predicted_prices[i].item(),
                        abs(predicted_prices[i].item() - true_prices_tensor[i].item()),
                        abs(predicted_prices[i].item() - true_prices_tensor[i].item()) / (true_prices_tensor[i].item() + 1e-8) * 100
                    ])
                
                table = wandb.Table(
                    data=comparison_data,
                    columns=["true_price", "predicted_price", "absolute_error", "percentage_error"]
                )
                
                wandb.log({
                    "predictions/comparison_table": table,
                }, step=self.state.global_step)
        
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            print(f"Error in loss computation: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, requires_grad=True, device=model.device)





# ==================== Checkpoint Resuming ====================

def find_latest_checkpoint(output_dir):
    """
    Find the latest checkpoint in the output directory.
    Returns None if no checkpoints exist.
    """
    if not os.path.exists(output_dir):
        return None
    
    # Find all checkpoint directories
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(output_dir, item)
            if os.path.isdir(checkpoint_path):
                try:
                    # Extract step number
                    step = int(item.split('-')[1])
                    checkpoints.append((step, checkpoint_path))
                except:
                    continue
    
    if not checkpoints:
        return None
    
    # Sort by step number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_path = checkpoints[0]
    
    return latest_path

def load_mlp_from_checkpoint(checkpoint_dir, device):
    """
    Load MLP head from a checkpoint directory.
    """
    mlp_path = os.path.join(checkpoint_dir, "mlp_head.pt")
    
    if not os.path.exists(mlp_path):
        raise FileNotFoundError(f"MLP head not found in checkpoint: {mlp_path}")
    
    mlp_checkpoint = torch.load(mlp_path, map_location=device)
    
    mlp_head = PriceRegressionHead(
        input_dim=mlp_checkpoint['config']['input_dim'],
        hidden_dims=mlp_checkpoint['config']['hidden_dims'],
        dropout=mlp_checkpoint['config']['dropout']
    )
    mlp_head.load_state_dict(mlp_checkpoint['state_dict'])
    mlp_head.to(device)
    
    return mlp_head

# ==================== Main Training ====================

def main():
    print("=" * 60)
    print("Product Pricing Prediction - Training with W&B")
    print("=" * 60)
    
    # Check for existing checkpoints
    resume_from_checkpoint = None
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    
    if latest_checkpoint:
        print(f"\n🔄 Found existing checkpoint: {latest_checkpoint}")
        response = input("Resume from this checkpoint? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            resume_from_checkpoint = latest_checkpoint
            print(f"✓ Will resume training from {latest_checkpoint}")
        else:
            print("✓ Starting fresh training")
    else:
        print("\n✓ No existing checkpoints found - starting fresh training")
    
    # Initialize W&B
    print("\nInitializing Weights & Biases...")
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        entity=WANDB_ENTITY,
        resume="allow" if resume_from_checkpoint else None,  # Allow resuming in W&B
        config={
            "model": "Qwen2.5-VL-7B-Instruct",
            "task": "product_pricing_prediction",
            "max_samples": MAX_SAMPLES,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "num_epochs": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "lora_r": 16,
            "lora_alpha": 16,
            "loss_function": "SmoothL1 on log(price)" if USE_LOG_TARGET else "Direct SMAPE",
            "use_log_target": USE_LOG_TARGET,
            "smape_epsilon": SMAPE_EPSILON,
            "mlp_input_dim": MLP_INPUT_DIM,
            "mlp_hidden_dims": MLP_HIDDEN_DIMS,
            "mlp_dropout": MLP_DROPOUT,
            "mlp_activation": "exp() for positive prices",
        },
        tags=["vision-language", "pricing", "qwen", "lora", "smape", "mlp-regression"],
    )
    
    # Load model
    print("\nLoading Qwen2.5-VL-7B model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Log model architecture summary
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Initialize MLP regression head
    print("\nInitializing MLP regression head...")
    
    if resume_from_checkpoint and os.path.exists(os.path.join(resume_from_checkpoint, "mlp_head.pt")):
        # Load MLP from checkpoint
        print(f"  Loading MLP from checkpoint...")
        mlp_head = load_mlp_from_checkpoint(resume_from_checkpoint, model.device)
        mlp_head.train()
        print(f"  ✓ MLP loaded from checkpoint")
    else:
        # Initialize new MLP
        mlp_head = PriceRegressionHead(
            input_dim=MLP_INPUT_DIM,
            hidden_dims=MLP_HIDDEN_DIMS,
            dropout=MLP_DROPOUT
        )
        mlp_head.to(model.device)
        mlp_head.train()
        print(f"  ✓ MLP initialized from scratch")
    
    # Verify MLP parameters are trainable
    mlp_params = sum(p.numel() for p in mlp_head.parameters())
    mlp_trainable = sum(p.numel() for p in mlp_head.parameters() if p.requires_grad)
    
    if mlp_params != mlp_trainable:
        raise RuntimeError(f"MLP parameters not trainable! {mlp_trainable}/{mlp_params}")
    total_trainable = lora_params + mlp_params
    
    wandb.config.update({
        "lora_parameters": lora_params,
        "mlp_parameters": mlp_params,
        "total_trainable_parameters": total_trainable,
        "total_parameters": total_params,
        "trainable_percentage": 100 * total_trainable / total_params,
    })
    
    print(f"\n{'='*60}")
    print(f"Parameter Summary:")
    print(f"{'='*60}")
    print(f"LoRA parameters:     {lora_params:,} ({lora_params/1e6:.2f}M)")
    print(f"MLP head parameters: {mlp_params:,} ({mlp_params/1e6:.2f}M)")
    print(f"Total trainable:     {total_trainable:,} ({total_trainable/1e6:.2f}M)")
    print(f"Total parameters:    {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable %:         {100*total_trainable/total_params:.4f}%")
    print(f"{'='*60}\n")
    
    # Prepare dataset with train/val split
    train_dataset, val_dataset = prepare_dataset_with_split(
        TRAIN_CSV, IMAGE_FOLDER, MAX_SAMPLES, EVAL_SPLIT
    )
    
    # Log dataset info
    wandb.config.update({
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    })
    
    # Create callbacks for tracking losses and saving MLP
    wandb_callback = WandbEvalCallback()
    text_logging_callback = TextFileLoggingCallback(
        log_file=os.path.join(OUTPUT_DIR, "training_log.txt")
    )
    mlp_checkpoint_callback = MLPCheckpointCallback(
        mlp_head=mlp_head,
        mlp_config={
            'input_dim': MLP_INPUT_DIM,
            'hidden_dims': MLP_HIDDEN_DIMS,
            'dropout': MLP_DROPOUT,
        }
    )
    
    # Create trainer with custom loss function
    print("\nSetting up trainer with W&B integration and MLP regression...")
    if USE_LOG_TARGET:
        print(f"Training Strategy: SmoothL1 Loss on log(price) [ADVANCED]")
        print(f"  → MLP outputs log(price), then exp() for actual price")
        print(f"  → More stable for wide price ranges")
        print(f"  → SMAPE computed for monitoring only")
    else:
        print(f"Training Strategy: Direct SMAPE Loss [STANDARD]")
        print(f"  → MLP outputs price directly via exp()")
        print(f"  → SMAPE epsilon: {SMAPE_EPSILON}")
    FastVisionModel.for_training(model)
    
    trainer = CustomPricingTrainer(
        model=model,
        tokenizer=tokenizer,
        mlp_head=mlp_head,  # Pass MLP head for price prediction
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[wandb_callback, text_logging_callback, mlp_checkpoint_callback],  # ✓ W&B + Text logging + MLP checkpoints
        smape_epsilon=SMAPE_EPSILON,  # Pass epsilon for SMAPE
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=100,  # Increased from 10 - MLP needs proper warmup
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            max_grad_norm=1.0,  # Gradient clipping to prevent explosion
            
            # Logging - W&B integration
            logging_steps=10,
            logging_first_step=True,
            report_to="wandb",  # ← Enable W&B reporting
            
            # Evaluation strategy
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            per_device_eval_batch_size=BATCH_SIZE,
            
            # Checkpoint saving
            save_strategy="steps",
            save_steps=EVAL_STEPS,
            save_total_limit=5,  # Keep all 5 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Optimizer (using custom optimizer from create_optimizer())
            # optim not specified - will use our custom AdamW with 2 learning rates
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            
            # Output
            output_dir=OUTPUT_DIR,
            
            # W&B specific
            run_name=WANDB_RUN_NAME,  # ← Set run name
            
            # Required for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_LENGTH,
        ),
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    # Log GPU info to W&B
    wandb.config.update({
        "gpu_name": gpu_stats.name,
        "gpu_memory_gb": max_memory,
    })
    
    print(f"\n{'='*60}")
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    print(f"{'='*60}\n")
    
    # Train
    print("\nStarting training with W&B tracking...")
    print(f"🔗 View training at: {wandb.run.get_url()}")
    print(f"Will evaluate every {EVAL_STEPS} steps")
    print(f"Will save checkpoint every {EVAL_STEPS} steps")
    print(f"  → Each checkpoint includes: LoRA adapters + MLP head")
    print(f"  → Training logs: {os.path.join(OUTPUT_DIR, 'training_log.txt')}")
    if resume_from_checkpoint:
        print(f"  → Resuming from: {resume_from_checkpoint}")
    print(f"Training for {NUM_TRAIN_EPOCHS} epochs\n")
    
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    # Log memory stats to W&B
    wandb.log({
        "memory/peak_reserved_gb": used_memory,
        "memory/peak_training_gb": used_memory_for_lora,
        "memory/peak_percentage": used_percentage,
        "memory/training_percentage": lora_percentage,
    })
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Time: {trainer_stats.metrics['train_runtime']:.2f} seconds ({trainer_stats.metrics['train_runtime']/60:.2f} minutes)")
    print(f"Peak reserved memory: {used_memory} GB ({used_percentage}% of max)")
    print(f"Peak reserved for training: {used_memory_for_lora} GB ({lora_percentage}% of max)")
    
    # Print loss summary
    if len(wandb_callback.eval_losses) > 0:
        print(f"\n{'='*60}")
        print("Loss Summary:")
        print(f"{'='*60}")
        print(f"Best Eval Loss: {min(wandb_callback.eval_losses):.4f}")
        print(f"Final Eval Loss: {wandb_callback.eval_losses[-1]:.4f}")
        if len(wandb_callback.train_losses) > 0:
            print(f"Final Train Loss: {wandb_callback.train_losses[-1]:.4f}")
        print(f"{'='*60}\n")
    
    # Save final model (LoRA + MLP)
    print(f"\nSaving models to {LORA_MODEL_DIR}...")
    
    # Save LoRA model
    model.save_pretrained(LORA_MODEL_DIR)
    tokenizer.save_pretrained(LORA_MODEL_DIR)
    print(f"✓ LoRA model saved")
    
    # Save MLP head
    mlp_path = os.path.join(LORA_MODEL_DIR, "mlp_head.pt")
    torch.save({
        'state_dict': mlp_head.state_dict(),
        'config': {
            'input_dim': MLP_INPUT_DIM,
            'hidden_dims': MLP_HIDDEN_DIMS,
            'dropout': MLP_DROPOUT,
        }
    }, mlp_path)
    print(f"✓ MLP head saved to {mlp_path}")
    
    # Log model as W&B artifact (optional but recommended)
    print("\nUploading models to W&B...")
    artifact = wandb.Artifact(
        name=f"pricing-model-{wandb.run.id}",
        type="model",
        description="Fine-tuned Qwen2.5-VL-7B with MLP head for product pricing",
        metadata={
            "best_eval_loss": wandb_callback.best_eval_loss,
            "epochs": NUM_TRAIN_EPOCHS,
            "samples": MAX_SAMPLES,
            "architecture": "Qwen2.5-VL-7B + LoRA + MLP",
            "loss_function": "SMAPE only",
        }
    )
    artifact.add_dir(LORA_MODEL_DIR)
    wandb.log_artifact(artifact)
    
    print("\n✅ Training complete! Best model saved and uploaded to W&B.")
    print(f"📁 LoRA Model: {LORA_MODEL_DIR}")
    print(f"📁 MLP Head: {mlp_path}")
    print(f"📁 Checkpoints: {OUTPUT_DIR}")
    print(f"🔗 W&B Dashboard: {wandb.run.get_url()}")
    print(f"\n💡 Model outputs prices with float32 precision (15+ decimal places)")
    
    # Finish W&B run
    wandb.finish()

if __name__ == "__main__":
    main()