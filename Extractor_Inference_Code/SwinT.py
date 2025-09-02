import torch
import numpy as np
from PIL import Image
from vit_pytorch import ViT
from torchvision import transforms
from transformers import SwinModel

"""


░██████╗░██╗░░░░░░░██╗██╗███╗░░██╗
██╔════╝░██║░░██╗░░██║██║████╗░██║
╚█████╗░░╚██╗████╗██╔╝██║██╔██╗██║
░╚═══██╗░░████╔═████║░██║██║╚████║
██████╔╝░░╚██╔╝░╚██╔╝░██║██║░╚███║
╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚══╝

// Ian Bezerra - 2025 //


"""

def swint_inference(image_paths):
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", output_hidden_states=True)
    
    model.to(device)

    # Transforming images as in the original file
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Process images in smaller batches
    batch_size = len(image_paths) // 10  # Adjust based on your GPU memory
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        input_tensor = []
        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            input_tensor.append(img)

        # Stacking tensors for current batch
        input_batch = torch.stack(input_tensor)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            outputs = model(input_batch)
            
            # Get the hidden states from the final layer
            # hidden_states contains embeddings from each layer, and we want the last one
            final_layer_features = outputs.hidden_states[-1]
            
            # Get the [CLS] token embedding or average over patch dimension
            features = final_layer_features.mean(dim=1)  # Shape: [batch_size, hidden_dim]
            
        # Move results to CPU immediately to free GPU memory
        features = features.cpu().numpy()
        all_features.append(features)
        
        # Clear some memory
        del input_batch, features
        torch.cuda.empty_cache()
    
    if len(all_features) > 1:
        features_batch = np.vstack(all_features)
    else:
        features_batch = all_features[0]
    
    print("Features batch shape:")
    print(features_batch.shape)
        
    return features_batch
