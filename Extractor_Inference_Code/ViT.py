
import torch
import numpy as np
from PIL import Image
from vit_pytorch import ViT
from torchvision import transforms

"""

██╗░░░██╗██╗████████╗
██║░░░██║██║╚══██╔══╝
╚██╗░██╔╝██║░░░██║░░░
░╚████╔╝░██║░░░██║░░░
░░╚██╔╝░░██║░░░██║░░░
░░░╚═╝░░░╚═╝░░░╚═╝░░░

// Ian Bezerra - 2025 //


"""


def vit_inference(image_paths):
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # init model
    image_size = 384
    #classes = utils.get_imagenet_classes()
    model = ViT(
        image_size = image_size,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    model.to(device)
    # Tranfomrando imagens como no arquiovo original
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Process images in smaller batches
    batch_size = len(image_paths) // 10  # Adjust based on your GPU memory
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Processando imagens do batch atual
        input_tensors = []
        for img_path in batch_paths:
            # Carregando e transformando cada imagem
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            input_tensors.append(img_tensor)

        # Empilhando os tensores do batch atual
        input_batch = torch.stack(input_tensors)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            features = model(input_batch)
            
        # Move results to CPU immediately to free GPU memory
        features = features.cpu().numpy()
        all_features.append(features)
        
        # Clear some memory
        del input_batch, features
        torch.cuda.empty_cache()
    
    features_batch = np.vstack(all_features)
    return features_batch
