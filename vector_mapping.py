import torch
import clip
from PIL import Image
import numpy as np

model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

image = preprocess(Image.open("flip.png")).unsqueeze(0).to(model.visual.conv1.weight.device)

with torch.no_grad():
    image_features = model.encode_image(image)

# Convert to numpy and save as .npy
np.save("image_features.npy", image_features.cpu().numpy())
