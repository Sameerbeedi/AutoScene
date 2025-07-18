import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from pathlib import Path
import json
import glob

class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.image_files = image_files
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load satellite image
        img_path = os.path.join(self.image_dir, img_name)
        if img_name.endswith('.png') or img_name.endswith('.jpg'):
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Handle other formats
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        
        # Load ground truth mask - For DeepGlobe format
        base_name = os.path.splitext(img_name)[0]
        
        # DeepGlobe specific naming: if image is "1054_sat.jpg", mask is "1054_mask.png"
        if '_sat' in base_name:
            mask_name = base_name.replace('_sat', '_mask') + '.png'
        else:
            # Fallback to other naming conventions
            possible_mask_names = [
                f"{base_name}_mask.png",
                f"{base_name}_gt.png",
                f"{base_name}_label.png",
                f"{base_name}.png",
            ]
            mask_name = None
            for possible_name in possible_mask_names:
                if os.path.exists(os.path.join(self.image_dir, possible_name)):
                    mask_name = possible_name
                    break
    
        # Load the mask
        if mask_name and os.path.exists(os.path.join(self.image_dir, mask_name)):
            label_path = os.path.join(self.image_dir, mask_name)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Warning: No mask found for {img_name}")
            label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize if needed (DeepGlobe images are usually large)
        target_size = (512, 512)
        image = cv2.resize(image, target_size)
        label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert RGB to grayscale for single channel input
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Convert mask to proper class indices
        # For road segmentation: 0=background, 1=road
        # Assuming mask is binary (0 or 255)
        label = (label > 127).astype(np.int64)  # Convert to binary classes
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label).long()
        
        return image, label

def get_deepglobe_files(data_dir):
    """Get DeepGlobe dataset files from a directory"""
    # Look for satellite images (not mask files)
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    all_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Only include satellite images, exclude mask files
                if '_sat' in file.lower():
                    # DeepGlobe satellite images - check if mask exists
                    base_name = os.path.splitext(file)[0]
                    mask_file = base_name.replace('_sat', '_mask') + '.png'
                    if os.path.exists(os.path.join(data_dir, mask_file)):
                        all_files.append(file)
                elif ('_mask' not in file.lower() and '_gt' not in file.lower() and '_label' not in file.lower()):
                    # Other satellite images - check if corresponding mask exists
                    base_name = os.path.splitext(file)[0]
                    mask_file = f"{base_name}_mask.png"
                    if os.path.exists(os.path.join(data_dir, mask_file)):
                        all_files.append(file)
    
    return sorted(all_files)

def find_latest_checkpoint(checkpoint_dir='checkpoints_mit'):
    """Find the latest checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoint_files[-1]

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load checkpoint and return starting epoch and losses"""
    print(f"📥 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        # Manually set scheduler to correct epoch
        start_epoch = checkpoint['epoch'] + 1
        for _ in range(start_epoch // 10):  # StepLR with step_size=10
            scheduler.step()
    
    # Extract training history
    train_losses = checkpoint.get('train_losses', [checkpoint.get('train_loss', 0.0)])
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"📊 Resuming from epoch: {start_epoch}")
    print(f"📊 Previous training loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"📊 Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    return start_epoch, train_losses

def train_model(model, train_loader, num_epochs=50, device='cuda', resume_from_checkpoint=True):
    """Train the segmentation model - training only, no validation"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    start_epoch = 0
    
    # Create checkpoints directory
    checkpoint_dir = 'checkpoints_mit'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Try to load from checkpoint if resuming
    if resume_from_checkpoint:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            start_epoch, train_losses = load_checkpoint(
                model, optimizer, scheduler, latest_checkpoint, device
            )
            print(f"🔄 Resuming training from epoch {start_epoch}")
        else:
            print("🆕 No checkpoint found, starting fresh training")
    else:
        print("🆕 Starting fresh training (ignoring any existing checkpoints)")
    
    print(f"🚀 Training on {len(train_loader)} batches per epoch")
    print(f"📊 Total training samples: {len(train_loader.dataset)}")
    print(f"📊 Training epochs: {start_epoch} to {num_epochs}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase only
        model.train()
        train_loss = 0.0
        
        epoch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(epoch_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{train_loss/(batch_idx+1):.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        print(f'\n📊 Epoch {epoch+1}/{num_epochs} Summary:')
        print(f'   Training Loss: {train_loss:.4f}')
        print(f'   Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        scheduler.step()
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # ← Added scheduler state
                'train_loss': train_loss,
                'train_losses': train_losses,  # ← Added full training history
                'learning_rate': scheduler.get_last_lr()[0]
            }, checkpoint_path)
            print(f'   💾 Checkpoint saved: {checkpoint_path}')
        
        # Save best model based on training loss
        if len(train_losses) == 1 or train_loss < min(train_losses[:-1]):
            best_model_path = f'{checkpoint_dir}/best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # ← Added scheduler state
                'train_loss': train_loss,
                'train_losses': train_losses,  # ← Added full training history
                'learning_rate': scheduler.get_last_lr()[0]
            }, best_model_path)
            print(f'   ⭐ Best model saved: {best_model_path}')
    
    return train_losses

def plot_training_history(train_losses, start_epoch=0):
    """Plot training loss curve only"""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Training loss plot
    plt.subplot(2, 2, 1)
    epochs = range(start_epoch + 1, start_epoch + len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add resume point indicator if training was resumed
    if start_epoch > 0:
        plt.axvline(x=start_epoch + 1, color='red', linestyle='--', alpha=0.7, label='Resume Point')
        plt.legend()
    
    # Loss improvement plot
    plt.subplot(2, 2, 2)
    if len(train_losses) > 1:
        loss_improvements = [train_losses[i-1] - train_losses[i] for i in range(1, len(train_losses))]
        improvement_epochs = range(start_epoch + 2, start_epoch + len(train_losses) + 1)
        plt.plot(improvement_epochs, loss_improvements, 'g-', 
                label='Loss Improvement', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Improvement')
        plt.title('Training Loss Improvement per Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Loss distribution
    plt.subplot(2, 2, 3)
    plt.hist(train_losses, bins=min(10, len(train_losses)), alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Loss Value')
    plt.ylabel('Frequency')
    plt.title('Training Loss Distribution')
    plt.grid(True, alpha=0.3)
    
    # Loss statistics
    plt.subplot(2, 2, 4)
    stats_text = f"""Training Statistics:
    
Initial Loss: {train_losses[0]:.4f}
Final Loss: {train_losses[-1]:.4f}
Best Loss: {min(train_losses):.4f}
Worst Loss: {max(train_losses):.4f}
Average Loss: {np.mean(train_losses):.4f}
Loss Std: {np.std(train_losses):.4f}

Start Epoch: {start_epoch + 1}
End Epoch: {start_epoch + len(train_losses)}
Total Epochs: {len(train_losses)}
Loss Reduction: {train_losses[0] - train_losses[-1]:.4f}
Improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%"""
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training plot saved: results/training_history.png")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Add command line option for resuming
    import sys
    resume_training = True  # Default to resume
    fresh_start = False
    
    # Simple command line argument parsing
    if len(sys.argv) > 1:
        if '--fresh' in sys.argv or '--no-resume' in sys.argv:
            fresh_start = True
            resume_training = False
            print("🆕 Fresh training requested - ignoring existing checkpoints")
        elif '--resume' in sys.argv:
            resume_training = True
            print("🔄 Resume training requested")
    
    # Dataset paths
    data_root = "D:/college/internship/PESURF/Satellite-Image-Road-Segmentation/Datasets/DeepGlobe"
    
    # Define train directory only
    train_dir = os.path.join(data_root, 'train')
    
    print(f"📁 Data root: {data_root}")
    print(f"📁 Train directory: {train_dir}")
    
    # Check if train directory exists
    if not os.path.exists(train_dir):
        print(f"❌ Train directory not found: {train_dir}")
        return
    
    # Get training files
    train_files = get_deepglobe_files(train_dir)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Training files: {len(train_files)}")
    
    # Check if sufficient files were found
    if len(train_files) == 0:
        print("❌ No training files found! Please check the dataset path.")
        return
    
    # Print some example files
    print(f"\n📋 Example training files:")
    for i, file in enumerate(train_files[:5]):
        print(f"   {i+1}. {file}")
    
    if len(train_files) > 5:
        print(f"   ... and {len(train_files) - 5} more files")
    
    # Create dataset - using same directory for images and masks
    train_dataset = DeepGlobeDataset(train_dir, train_files)
    
    print(f"\n📊 Dataset size: {len(train_dataset)} samples")
    
    # Create data loader
    batch_size = 1  # Small batch size for large images
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"📦 Training batches: {len(train_loader)}")
    print(f"📦 Batch size: {batch_size}")
    
    # Initialize model
    model = smp.Unet(
        encoder_name="mit_b5",        # Pre-trained encoder
        encoder_weights="imagenet",     # Pre-trained weights
        in_channels=1,                  # Grayscale input
        classes=2,                      # Binary segmentation: background (0), road (1)
    )
    
    model = model.to(device)
    print(f"🧠 Model initialized successfully!")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check for existing checkpoints
    if resume_training:
        latest_checkpoint = find_latest_checkpoint('checkpoints_mit')
        if latest_checkpoint:
            print(f"📁 Found existing checkpoint: {latest_checkpoint}")
            print("🔄 Training will resume from this checkpoint")
        else:
            print("📁 No existing checkpoints found")
            print("🆕 Will start fresh training")
    
    # Train the model
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING (TRAIN ONLY - NO VALIDATION)")
    print("="*60)
    
    train_losses = train_model(model, train_loader, num_epochs=50, device=device, 
                              resume_from_checkpoint=resume_training)
    
    # Calculate start epoch for plotting
    start_epoch = 0
    if resume_training:
        latest_checkpoint = find_latest_checkpoint('checkpoints_mit')
        if latest_checkpoint:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            start_epoch = checkpoint['epoch'] + 1 - len(train_losses)
    
    # Plot training history
    plot_training_history(train_losses, start_epoch)
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    final_model_path = 'models/final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\n💾 Final model saved as '{final_model_path}'")
    
    # Save training summary
    summary = {
        'training_mode': 'train_only',
        'validation_used': False,
        'resumed_from_checkpoint': resume_training and find_latest_checkpoint('checkpoints_mit') is not None,
        'train_files': len(train_files),
        'total_samples': len(train_dataset),
        'epochs_trained': len(train_losses),
        'total_epochs': 40,
        'batch_size': batch_size,
        'initial_train_loss': train_losses[0],
        'final_train_loss': train_losses[-1],
        'best_train_loss': min(train_losses),
        'loss_improvement': train_losses[0] - train_losses[-1],
        'loss_improvement_percent': ((train_losses[0] - train_losses[-1]) / train_losses[0] * 100),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'device': str(device)
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Final training loss: {train_losses[-1]:.4f}")
    print(f"📊 Best training loss: {min(train_losses):.4f}")
    print(f"📊 Loss improvement: {train_losses[0] - train_losses[-1]:.4f} ({((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%)")
    print(f"📁 Results saved in 'results/' directory")
    print(f"📁 Checkpoints saved in 'checkpoints_mit/' directory")
    print(f"📁 Models saved in 'models/' directory")

if __name__ == "__main__":
    main()
