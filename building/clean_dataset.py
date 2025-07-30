import os
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def check_mask_has_buildings(mask_path):
    """Check if a mask has any building pixels (non-zero values)"""
    try:
        # Try loading with cv2 first
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Fallback to PIL
            mask_pil = Image.open(mask_path)
            if mask_pil.mode != 'L':
                mask_pil = mask_pil.convert('L')
            mask = np.array(mask_pil)
        
        # Check if there are any non-zero pixels (buildings)
        building_pixels = np.sum(mask > 0)
        return building_pixels > 0, building_pixels
    
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return False, 0

def get_image_mask_pairs(image_dir, label_dir):
    """Get pairs of image and mask files"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    pairs = []
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        return pairs
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            base_name = os.path.splitext(file)[0]
            if '_vis' not in base_name:  # Skip visualization files
                image_path = os.path.join(image_dir, file)
                mask_path = os.path.join(label_dir, f"{base_name}.tif")
                
                if os.path.exists(mask_path):
                    pairs.append((image_path, mask_path, file, f"{base_name}.tif"))
    
    return pairs

def clean_dataset(data_root, move_empty=True, delete_empty=False):
    """
    Clean dataset by moving or deleting images with empty masks
    
    Args:
        data_root: Root directory containing train/val/test folders
        move_empty: If True, move empty mask files to 'empty_masks' folder
        delete_empty: If True, delete empty mask files (use with caution)
    """
    
    splits = ['train', 'val', 'test']
    total_moved = 0
    total_kept = 0
    
    for split in splits:
        image_dir = os.path.join(data_root, split, 'image')
        label_dir = os.path.join(data_root, split, 'label')
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"⚠️  Skipping {split} - directories not found")
            continue
        
        print(f"\n🔍 Processing {split} split...")
        
        # Get all image-mask pairs
        pairs = get_image_mask_pairs(image_dir, label_dir)
        print(f"Found {len(pairs)} image-mask pairs in {split}")
        
        if len(pairs) == 0:
            continue
        
        # Create directories for empty masks if moving
        if move_empty:
            empty_image_dir = os.path.join(data_root, split, 'empty_masks', 'image')
            empty_label_dir = os.path.join(data_root, split, 'empty_masks', 'label')
            os.makedirs(empty_image_dir, exist_ok=True)
            os.makedirs(empty_label_dir, exist_ok=True)
        
        empty_count = 0
        kept_count = 0
        
        # Process each pair
        for image_path, mask_path, image_file, mask_file in tqdm(pairs, desc=f"Checking {split}"):
            has_buildings, building_pixels = check_mask_has_buildings(mask_path)
            
            if not has_buildings:
                # Empty mask found
                empty_count += 1
                print(f"   📄 Empty mask: {image_file} (0 building pixels)")
                
                if delete_empty:
                    # Delete both image and mask
                    os.remove(image_path)
                    os.remove(mask_path)
                    print(f"   🗑️  Deleted: {image_file} and {mask_file}")
                
                elif move_empty:
                    # Move both image and mask to empty_masks folder
                    new_image_path = os.path.join(empty_image_dir, image_file)
                    new_mask_path = os.path.join(empty_label_dir, mask_file)
                    
                    shutil.move(image_path, new_image_path)
                    shutil.move(mask_path, new_mask_path)
                    print(f"   📦 Moved: {image_file} and {mask_file} to empty_masks/")
            
            else:
                # Mask has buildings, keep it
                kept_count += 1
        
        print(f"\n📊 {split.upper()} Summary:")
        print(f"   ✅ Kept: {kept_count} pairs (with buildings)")
        print(f"   {'🗑️  Deleted' if delete_empty else '📦 Moved'}: {empty_count} pairs (empty masks)")
        
        total_moved += empty_count
        total_kept += kept_count
    
    print(f"\n🎉 Dataset Cleaning Complete!")
    print(f"   ✅ Total kept: {total_kept} image-mask pairs")
    print(f"   {'🗑️  Total deleted' if delete_empty else '📦 Total moved'}: {total_moved} pairs")
    
    if move_empty and not delete_empty:
        print(f"\n📁 Empty mask files moved to respective 'empty_masks' folders")
        print(f"   You can review these files and delete them later if needed")
    
    return total_kept, total_moved

def main():
    # Dataset root directory
    data_root = os.path.join(os.getcwd(), "9")
    
    print("🧹 Dataset Cleaning Tool")
    print("="*50)
    print(f"📁 Dataset root: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"❌ Dataset root not found: {data_root}")
        return
    
    print("\nChoose an action:")
    print("1. Move empty mask files to 'empty_masks' folder (SAFE - recommended)")
    print("2. Delete empty mask files permanently (DANGEROUS - cannot be undone)")
    print("3. Just count empty masks (no changes)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\n🚀 Moving empty mask files...")
        clean_dataset(data_root, move_empty=True, delete_empty=False)
    
    elif choice == '2':
        confirm = input("\n⚠️  Are you SURE you want to DELETE files? Type 'DELETE' to confirm: ")
        if confirm == 'DELETE':
            print("\n🗑️  Deleting empty mask files...")
            clean_dataset(data_root, move_empty=False, delete_empty=True)
        else:
            print("❌ Deletion cancelled")
    
    elif choice == '3':
        print("\n🔍 Counting empty masks (no changes will be made)...")
        
        splits = ['train', 'val', 'test']
        total_empty = 0
        total_files = 0
        
        for split in splits:
            image_dir = os.path.join(data_root, split, 'image')
            label_dir = os.path.join(data_root, split, 'label')
            
            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                continue
            
            pairs = get_image_mask_pairs(image_dir, label_dir)
            empty_count = 0
            
            for image_path, mask_path, image_file, mask_file in tqdm(pairs, desc=f"Checking {split}"):
                has_buildings, building_pixels = check_mask_has_buildings(mask_path)
                if not has_buildings:
                    empty_count += 1
                    print(f"   📄 Empty: {image_file}")
            
            print(f"\n📊 {split.upper()}: {empty_count} empty / {len(pairs)} total")
            total_empty += empty_count
            total_files += len(pairs)
        
        print(f"\n📊 OVERALL: {total_empty} empty masks / {total_files} total files")
        print(f"   Empty percentage: {(total_empty/total_files*100):.1f}%" if total_files > 0 else "")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
