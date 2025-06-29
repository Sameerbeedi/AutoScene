import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Paths
train_root_dir = "9/train/label"  
val_root_dir = "9/val/label"  
test_root_dir = "9/test/label"  
output_base_dir = "arieal_segmentation"  

# Function to process labels
def process_labels(root_dir, output_base_dir):
    # Iterate over files directly if no subdirectories exist
    label_files = [f for f in sorted(os.listdir(root_dir)) if f.endswith(".tif")]
    output_dir = output_base_dir
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Process each label file
    for fname in tqdm(label_files, desc=f"Converting .tif to .npy in {root_dir}"):
        label_path = os.path.join(root_dir, fname)
        label = Image.open(label_path)

        # Resize label image to manageable size
        label = label.resize((512, 512), Image.NEAREST) 
        label_array = np.array(label)  # Convert label image to numpy array

        # Save as .npy file
        npy_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.npy")
        np.save(npy_path, label_array)

# Process train, val, and test directories
process_labels(train_root_dir, os.path.join(output_base_dir, "train"))
process_labels(val_root_dir, os.path.join(output_base_dir, "val"))
process_labels(test_root_dir, os.path.join(output_base_dir, "test"))
