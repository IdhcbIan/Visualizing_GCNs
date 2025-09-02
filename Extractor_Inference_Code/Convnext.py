
import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import ConvNextImageProcessor, ConvNextForImageClassification


"""
░█████╗░░█████╗░███╗░░██╗██╗░░░██╗███╗░░██╗███████╗██╗░░██╗████████╗
██╔══██╗██╔══██╗████╗░██║██║░░░██║████╗░██║██╔════╝╚██╗██╔╝╚══██╔══╝
██║░░╚═╝██║░░██║██╔██╗██║╚██╗░██╔╝██╔██╗██║█████╗░░░╚███╔╝░░░░██║░░░
██║░░██╗██║░░██║██║╚████║░╚████╔╝░██║╚████║██╔══╝░░░██╔██╗░░░░██║░░░
╚█████╔╝╚█████╔╝██║░╚███║░░╚██╔╝░░██║░╚███║███████╗██╔╝╚██╗░░░██║░░░
░╚════╝░░╚════╝░╚═╝░░╚══╝░░░╚═╝░░░╚═╝░░╚══╝╚══════╝╚═╝░░╚═╝░░░╚═╝░░░

// Ian Bezerra - 2025 //

"""

def aloi_convnext_inference(version, image_paths):

    model_type = (version)

    print(f"\Extracting {model_type} embeddings for all images...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "ConvNext-tiny-224":
        feature_extractor = ConvNextImageProcessor.from_pretrained(
            "facebook/convnext-tiny-224")

        model = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-tiny-224").to(device)
        
    elif model_type == "ConvNext-large-224-22k-1k":
        feature_extractor = ConvNextImageProcessor.from_pretrained(
            "facebook/convnext-large-224-22k-1k")

        model = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-large-224-22k-1k").to(device)

    model.classifier = torch.nn.Identity()  # Removing softmax layer
    model.eval()  # Set model to evaluation mode

    features_list = []
    file_list = []
    
    # Get all image filenames
    batch_size = len(image_paths) // 10
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_images = []
        batch_file_names = []
        
        # Collect images for current batch
        for j in range(i, min(i + batch_size, len(image_paths))):
            image_name = image_paths[j]
            file_list.append(image_name)
            batch_file_names.append(image_name)
            
            # Load and convert image
            image = Image.open(f"{image_name}").convert("RGB")
            batch_images.append(image)
        
        # Process batch at once
        inputs = feature_extractor(batch_images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_features = outputs.logits.cpu().numpy()
            
            for features in batch_features:
                features_list.append(features.flatten())
                

    # Convert features to numpy array
    features_matrix = np.array(features_list)
    return features_matrix