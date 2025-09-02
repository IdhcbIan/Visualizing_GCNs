import os, sys
import numpy as np
import natsort
import tensorflow as tf
from vit_keras import vit, utils

"""

██╗░░░██╗██╗████████╗
██║░░░██║██║╚══██╔══╝
╚██╗░██╔╝██║░░░██║░░░
░╚████╔╝░██║░░░██║░░░
░░╚██╔╝░░██║░░░██║░░░
░░░╚═╝░░░╚═╝░░░╚═╝░░░

// Ian Bezerra - 2025 //

gpu still not working normal eta(flowers 15Min!!!)

"""

def vit_inference(image_paths):
    # init model
    image_size = 384
    classes = utils.get_imagenet_classes()
    model = vit.vit_b16(
        image_size=image_size,
        activation="sigmoid",
        pretrained=True,
        include_top=True,
        pretrained_top=True,
    )

    imgs_full = image_paths

    features = []
    # Create Runs directory if it doesn't exist
    
    #f = open("../Runs/ViT_images.txt", "w")
    
    # Process all images at once
    images = [utils.read(img_path, image_size) for img_path in imgs_full]

    # Limitando para as 100 primeiras imagens
    images = images[:100] 
   
    # Write all paths
    #f.write("\n".join(imgs_full))
    
    # Preprocess batch of images
    X = np.stack([vit.preprocess_inputs(img) for img in images])
    X = X.reshape(-1, image_size, image_size, 3)

    # Convert batch of preprocessed images to tensor
    image_tensor = tf.convert_to_tensor(X)

    print(image_tensor.shape)
    print("All images pre-processed")

    # Predict the image
    y = model.predict(image_tensor)
    print(f"y shape: {y.shape}")

    # Append the features
    features = y
    print(f"Features shape: {features.shape}")
    #f.close()

    return features