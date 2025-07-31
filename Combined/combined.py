import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path to import the dynamic ensemble
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import the original DynamicCheckpointEnsemble
import importlib.util
spec = importlib.util.spec_from_file_location("dynamic_checkpoint_ensemble", 
                                              os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                          "dynamic_checkpoint_ensemble.py"))
dce_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dce_module)

class CustomDynamicCheckpointEnsemble(dce_module.DynamicCheckpointEnsemble):
    """Custom version that handles correct path resolution"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        self.checkpoint_info = {}
        
        # Get the parent PESURF directory
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Define checkpoint directories with correct paths
        self.checkpoint_dirs = {
            'resnet': os.path.join(parent_dir, 'checkpoints'),
            'efficientnet': os.path.join(parent_dir, 'checkpoints_efficientNet'), 
            'mobilenet': os.path.join(parent_dir, 'checkpoints_mobilenet'),
            'mit': os.path.join(parent_dir, 'checkpoints_mit')
        }
        
        # Initialize all models and checkpoints
        self._discover_and_load_checkpoints()

class BuildingEnsemble:
    """Building segmentation ensemble using building checkpoints"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        # Use absolute path to building checkpoints
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.checkpoint_dir = os.path.join(parent_dir, 'building', 'checkpoints_building')
        
        # Initialize building models
        self._load_building_models()
        
    def _load_building_models(self):
        """Load all building checkpoint models"""
        print("🏢 Loading building models...")
        
        if not os.path.exists(self.checkpoint_dir):
            print(f"❌ Building checkpoint directory not found: {self.checkpoint_dir}")
            return
        
        # Find all checkpoint files
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "*.pth"))
        
        if not checkpoint_files:
            print(f"❌ No building checkpoints found in {self.checkpoint_dir}")
            return
        
        for checkpoint_path in checkpoint_files:
            checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '')
            
            try:
                # Create U-Net model for building segmentation
                model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights=None,
                    in_channels=1,
                    classes=2  # background and building
                ).to(self.device)
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                self.models[checkpoint_name] = model
                print(f"✅ Loaded building model: {checkpoint_name}")
                
            except Exception as e:
                print(f"❌ Failed to load building model {checkpoint_name}: {e}")
        
        print(f"🏢 Total building models loaded: {len(self.models)}")
    
    def preprocess_image(self, image_path, target_size=(512, 512)):
        """Preprocess image for building model input"""
        if isinstance(image_path, str):
            # Handle different image formats
            if image_path.lower().endswith(('.tif', '.tiff')):
                try:
                    from rasterio import open as rio_open
                    with rio_open(image_path) as src:
                        image = src.read(1)
                except:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path
            
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Resize and normalize
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        return image.to(self.device)
    
    def predict_buildings(self, image_tensor):
        """Get building predictions from all models"""
        predictions = {}
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                try:
                    output = model(image_tensor)
                    # Apply softmax to get probabilities
                    prob = F.softmax(output, dim=1)
                    # Extract building probability (class 1)
                    building_prob = prob[:, 1:2, :, :]
                    predictions[model_name] = building_prob
                except Exception as e:
                    print(f"❌ Error in building prediction with {model_name}: {e}")
        
        return predictions
    
    def ensemble_buildings(self, predictions):
        """Create ensemble prediction for buildings"""
        if not predictions:
            return None
        
        # Simple average ensemble
        stacked_preds = torch.stack(list(predictions.values()))
        ensemble_pred = torch.mean(stacked_preds, dim=0)
        
        return ensemble_pred

class CombinedRoadBuildingPredictor:
    """Combined predictor for roads and buildings using parallel processing"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize road ensemble (dynamic checkpoint)
        print("🛣️ Initializing road prediction ensemble...")
        self.road_ensemble = CustomDynamicCheckpointEnsemble(device=device)
        
        # Initialize building ensemble
        print("🏢 Initializing building prediction ensemble...")
        self.building_ensemble = BuildingEnsemble(device=device)
        
        # Color mapping for visualization
        self.colors = {
            'background': [0, 0, 0],      # Black
            'roads': [0, 255, 0],         # Green
            'buildings': [255, 0, 0],     # Red
            'overlap': [255, 255, 0]      # Yellow (for overlap regions)
        }
    
    def predict_parallel(self, image_path, road_strategy='balanced', road_top_k=6, 
                        building_threshold=0.5, road_threshold=0.5, verbose=True):
        """Predict roads and buildings in parallel"""
        
        if verbose:
            print(f"🔄 Processing: {os.path.basename(image_path)}")
        
        # Shared results dictionary
        results = {'road': None, 'building': None, 'errors': []}
        lock = threading.Lock()
        
        def predict_roads():
            """Road prediction thread"""
            try:
                if verbose:
                    print("🛣️ Starting road prediction...")
                
                road_pred, _, selected_checkpoints, weights = self.road_ensemble.predict_single(
                    image_path, 
                    strategy=road_strategy, 
                    top_k=road_top_k, 
                    verbose=False
                )
                
                # Convert to binary mask
                road_prob = torch.sigmoid(road_pred).cpu().numpy().squeeze()
                road_mask = (road_prob > road_threshold).astype(np.uint8)
                
                with lock:
                    results['road'] = {
                        'mask': road_mask,
                        'probability': road_prob,
                        'selected_checkpoints': selected_checkpoints,
                        'weights': weights
                    }
                
                if verbose:
                    print("✅ Road prediction completed")
                    
            except Exception as e:
                error_msg = f"Road prediction error: {e}"
                print(f"❌ {error_msg}")
                with lock:
                    results['errors'].append(error_msg)
        
        def predict_buildings():
            """Building prediction thread"""
            try:
                if verbose:
                    print("🏢 Starting building prediction...")
                
                # Preprocess image for building model
                building_tensor = self.building_ensemble.preprocess_image(image_path)
                
                # Get building predictions
                building_predictions = self.building_ensemble.predict_buildings(building_tensor)
                
                if building_predictions:
                    # Create ensemble
                    building_ensemble = self.building_ensemble.ensemble_buildings(building_predictions)
                    
                    # Convert to binary mask
                    building_prob = building_ensemble.cpu().numpy().squeeze()
                    building_mask = (building_prob > building_threshold).astype(np.uint8)
                    
                    with lock:
                        results['building'] = {
                            'mask': building_mask,
                            'probability': building_prob,
                            'model_count': len(building_predictions)
                        }
                else:
                    with lock:
                        results['errors'].append("No building predictions generated")
                
                if verbose:
                    print("✅ Building prediction completed")
                    
            except Exception as e:
                error_msg = f"Building prediction error: {e}"
                print(f"❌ {error_msg}")
                with lock:
                    results['errors'].append(error_msg)
        
        # Run predictions in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            road_future = executor.submit(predict_roads)
            building_future = executor.submit(predict_buildings)
            
            # Wait for both to complete
            road_future.result()
            building_future.result()
        
        return results
    
    def create_combined_visualization(self, results, image_path, save_path=None):
        """Create combined visualization with different colors for roads and buildings"""
        
        # Load original image for visualization
        try:
            if image_path.lower().endswith(('.tif', '.tiff')):
                try:
                    from rasterio import open as rio_open
                    with rio_open(image_path) as src:
                        original = src.read(1)
                except:
                    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if original is None:
                raise ValueError("Could not load original image")
                
            # Resize to match prediction size
            original = cv2.resize(original, (512, 512))
            
        except Exception as e:
            print(f"⚠️ Could not load original image: {e}")
            original = np.zeros((512, 512), dtype=np.uint8)
        
        # Create combined mask
        height, width = 512, 512
        combined_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background (original image in grayscale)
        for i in range(3):
            combined_mask[:, :, i] = original
        
        # Add roads (green)
        if results['road'] is not None:
            road_mask = results['road']['mask']
            road_indices = road_mask > 0
            combined_mask[road_indices] = self.colors['roads']
        
        # Add buildings (red)
        if results['building'] is not None:
            building_mask = results['building']['mask']
            building_indices = building_mask > 0
            combined_mask[building_indices] = self.colors['buildings']
        
        # Handle overlap (roads and buildings) - yellow
        if results['road'] is not None and results['building'] is not None:
            road_mask = results['road']['mask']
            building_mask = results['building']['mask']
            overlap_indices = (road_mask > 0) & (building_mask > 0)
            combined_mask[overlap_indices] = self.colors['overlap']
        
        # Create detailed visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Road mask
        if results['road'] is not None:
            axes[0, 1].imshow(results['road']['mask'], cmap='Greens', vmin=0, vmax=1)
            axes[0, 1].set_title('Roads (Green)')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Road\nPrediction', ha='center', va='center',
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Roads (Error)')
            axes[0, 1].axis('off')
        
        # Building mask
        if results['building'] is not None:
            axes[0, 2].imshow(results['building']['mask'], cmap='Reds', vmin=0, vmax=1)
            axes[0, 2].set_title('Buildings (Red)')
            axes[0, 2].axis('off')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Building\nPrediction', ha='center', va='center',
                           transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Buildings (Error)')
            axes[0, 2].axis('off')
        
        # Combined result
        axes[1, 0].imshow(combined_mask)
        axes[1, 0].set_title('Combined Result\n(Green: Roads, Red: Buildings, Yellow: Overlap)')
        axes[1, 0].axis('off')
        
        # Road probability map
        if results['road'] is not None:
            im1 = axes[1, 1].imshow(results['road']['probability'], cmap='viridis', vmin=0, vmax=1)
            axes[1, 1].set_title('Road Probability Map')
            axes[1, 1].axis('off')
            plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        else:
            axes[1, 1].axis('off')
        
        # Building probability map
        if results['building'] is not None:
            im2 = axes[1, 2].imshow(results['building']['probability'], cmap='plasma', vmin=0, vmax=1)
            axes[1, 2].set_title('Building Probability Map')
            axes[1, 2].axis('off')
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        return fig, combined_mask
    
    def process_single_image(self, image_path, save_dir=None, **kwargs):
        """Process a single image and create comprehensive results"""
        
        image_name = Path(image_path).stem
        
        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions
        print(f"🔄 Processing {image_name}...")
        results = self.predict_parallel(image_path, **kwargs)
        
        # Check for errors
        if results['errors']:
            print(f"⚠️ Errors occurred during processing:")
            for error in results['errors']:
                print(f"   - {error}")
        
        # Create visualization
        if save_dir:
            viz_path = os.path.join(save_dir, f"{image_name}_combined_result.png")
        else:
            viz_path = None
        
        fig, combined_mask = self.create_combined_visualization(results, image_path, viz_path)
        
        # Save individual masks
        if save_dir:
            if results['road'] is not None:
                road_path = os.path.join(save_dir, f"{image_name}_roads.png")
                cv2.imwrite(road_path, (results['road']['mask'] * 255).astype(np.uint8))
            
            if results['building'] is not None:
                building_path = os.path.join(save_dir, f"{image_name}_buildings.png")
                cv2.imwrite(building_path, (results['building']['mask'] * 255).astype(np.uint8))
            
            # Save combined mask
            combined_path = os.path.join(save_dir, f"{image_name}_combined_mask.png")
            cv2.imwrite(combined_path, cv2.cvtColor(combined_mask, cv2.COLOR_RGB2BGR))
            
            # Save detailed results JSON
            result_data = {
                'image_path': image_path,
                'errors': results['errors'],
                'statistics': {}
            }
            
            if results['road'] is not None:
                result_data['road'] = {
                    'road_coverage_percentage': float(np.mean(results['road']['mask']) * 100),
                    'mean_probability': float(np.mean(results['road']['probability'])),
                    'max_probability': float(np.max(results['road']['probability'])),
                    'selected_checkpoints': [
                        {'model_type': mt, 'checkpoint': cn, 'score': float(sc)}
                        for sc, mt, cn in results['road']['selected_checkpoints']
                    ],
                    'weights': {k: float(v) for k, v in results['road']['weights'].items()}
                }
            
            if results['building'] is not None:
                result_data['building'] = {
                    'building_coverage_percentage': float(np.mean(results['building']['mask']) * 100),
                    'mean_probability': float(np.mean(results['building']['probability'])),
                    'max_probability': float(np.max(results['building']['probability'])),
                    'model_count': results['building']['model_count']
                }
            
            # Calculate overlap statistics
            if results['road'] is not None and results['building'] is not None:
                overlap = (results['road']['mask'] > 0) & (results['building']['mask'] > 0)
                result_data['statistics'] = {
                    'total_pixels': int(np.prod(results['road']['mask'].shape)),
                    'road_pixels': int(np.sum(results['road']['mask'])),
                    'building_pixels': int(np.sum(results['building']['mask'])),
                    'overlap_pixels': int(np.sum(overlap)),
                    'overlap_percentage': float(np.sum(overlap) / max(1, np.sum(results['road']['mask'])) * 100)
                }
            
            json_path = os.path.join(save_dir, f"{image_name}_results.json")
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
        
        return results, fig, combined_mask
    
    def process_batch(self, image_paths, save_dir='combined_results', **kwargs):
        """Process multiple images"""
        
        if isinstance(image_paths, str):
            original_path = image_paths
            if os.path.isfile(original_path):
                image_paths = [original_path]
            elif os.path.isdir(original_path):
                # Find all images in directory
                image_paths = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                    image_paths.extend(glob.glob(os.path.join(original_path, ext)))
            else:
                # Assume glob pattern
                image_paths = glob.glob(original_path)
        
        # Filter out result files
        image_paths = [img for img in image_paths 
                      if not any(keyword in img.lower() 
                               for keyword in ['result', 'ensemble', 'combined', 'comparison'])]
        
        if not image_paths:
            print("❌ No valid images found")
            return []
        
        print(f"📷 Processing {len(image_paths)} images...")
        
        results = []
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                result, fig, combined_mask = self.process_single_image(
                    image_path, save_dir=save_dir, verbose=False, **kwargs
                )
                results.append({
                    'image_path': image_path,
                    'result': result,
                    'success': True
                })
                plt.close(fig)  # Close figure to save memory
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
        
        # Create batch summary
        successful = [r for r in results if r['success']]
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total images: {len(image_paths)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(image_paths) - len(successful)}")
        print(f"   Results saved to: {save_dir}")
        
        return results

def main():
    """Example usage of the combined road and building predictor"""
    print("🚀 Combined Road & Building Segmentation System")
    print("=" * 60)
    
    # Initialize the combined predictor
    predictor = CombinedRoadBuildingPredictor()
    
    # Get user input
    print("\n📁 Please specify image path(s):")
    print("1. Single image file")
    print("2. Directory containing images")
    print("3. Glob pattern (e.g., *.jpg)")
    print("4. Use demo images from current directory")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        image_path = input("Enter image path: ").strip().strip('"\'')
        if not os.path.exists(image_path):
            print("❌ Image not found")
            return
        image_paths = [image_path]
        
    elif choice == '2':
        dir_path = input("Enter directory path: ").strip().strip('"\'')
        if not os.path.isdir(dir_path):
            print("❌ Directory not found")
            return
        image_paths = dir_path
        
    elif choice == '3':
        pattern = input("Enter glob pattern: ").strip().strip('"\'')
        image_paths = glob.glob(pattern)
        if not image_paths:
            print("❌ No images found matching pattern")
            return
            
    elif choice == '4':
        # Find demo images
        demo_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
            demo_images.extend(glob.glob(ext))
        
        # Filter out result images
        demo_images = [img for img in demo_images 
                      if not any(keyword in img.lower() 
                               for keyword in ['result', 'ensemble', 'combined'])]
        
        if not demo_images:
            print("❌ No demo images found in current directory")
            return
        
        image_paths = demo_images[:3]  # Limit to first 3 for demo
        print(f"📷 Using demo images: {[os.path.basename(p) for p in image_paths]}")
        
    else:
        print("❌ Invalid choice")
        return
    
    # Get prediction settings
    print("\n⚙️ Prediction Settings:")
    road_strategy = input("Road strategy (balanced/confidence/agreement/diversity) [balanced]: ").strip() or 'balanced'
    
    try:
        road_top_k = int(input("Number of road checkpoints [6]: ").strip() or '6')
    except ValueError:
        road_top_k = 6
    
    try:
        road_threshold = float(input("Road threshold [0.5]: ").strip() or '0.5')
    except ValueError:
        road_threshold = 0.5
    
    try:
        building_threshold = float(input("Building threshold [0.5]: ").strip() or '0.5')
    except ValueError:
        building_threshold = 0.5
    
    save_dir = input("Save directory [combined_results]: ").strip() or 'combined_results'
    
    print(f"\n📋 Configuration:")
    print(f"   Road strategy: {road_strategy}")
    print(f"   Road checkpoints: {road_top_k}")
    print(f"   Road threshold: {road_threshold}")
    print(f"   Building threshold: {building_threshold}")
    print(f"   Save directory: {save_dir}")
    
    # Process images
    if isinstance(image_paths, list) and len(image_paths) == 1:
        # Single image processing with detailed output
        results, fig, combined_mask = predictor.process_single_image(
            image_paths[0],
            save_dir=save_dir,
            road_strategy=road_strategy,
            road_top_k=road_top_k,
            road_threshold=road_threshold,
            building_threshold=building_threshold
        )
        
        plt.show()
        print(f"✅ Results saved to {save_dir}")
        
    else:
        # Batch processing
        results = predictor.process_batch(
            image_paths,
            save_dir=save_dir,
            road_strategy=road_strategy,
            road_top_k=road_top_k,
            road_threshold=road_threshold,
            building_threshold=building_threshold
        )
        
        print(f"✅ Batch processing completed. Results saved to {save_dir}")

if __name__ == "__main__":
    main()
