import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# DinoV3 imports
from transformers import AutoModel
from huggingface_hub import login


"""

██████╗░██╗███╗░░██╗░█████╗░██╗░░░██╗██████╗░
██╔══██╗██║████╗░██║██╔══██╗██║░░░██║╚════██╗
██║░░██║██║██╔██╗██║██║░░██║╚██╗░██╔╝░█████╔╝
██║░░██║██║██║╚████║██║░░██║░╚████╔╝░░╚═══██╗
██████╔╝██║██║░╚███║╚█████╔╝░░╚██╔╝░░██████╔╝
╚═════╝░╚═╝╚═╝░░╚══╝░╚════╝░░░░╚═╝░░░╚═════╝░

// Ian Bezerra - 2025 //

    Usando o codigo base apra criar uma funcao que extrai features(inferencia) em batch!!

"""

from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv('HUGGING_FACE_TOKEN') # O Modelo DinoV3 Prescisa de autenticacao no HuggingFace


if hf_token:
    login(token=hf_token)
    print("Successfully authenticated with HuggingFace")
else:
    print("Warning: HF_TOKEN not found, model access may be restricted")


def dinov3_inference(model_name, image_paths):
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading model
    full_model_name =  "facebook/" + model_name
    dinov3 = AutoModel.from_pretrained(full_model_name)
    dinov3.to(device)
    dinov3.eval()


    # Tranfomrando imagens como no arquiovo original
    transform = transforms.Compose([
        transforms.Resize(256),
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
            out = dinov3(input_batch)
            features = out.last_hidden_state[:, 0, :]
            
        # Move results to CPU immediately to free GPU memory
        features = features.cpu().numpy()
        all_features.append(features)
        
        # Clear some memory
        del input_batch, features
        torch.cuda.empty_cache()
    
    features_batch = np.vstack(all_features)
    return features_batch
