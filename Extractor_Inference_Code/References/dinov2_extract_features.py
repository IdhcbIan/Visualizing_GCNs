import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import natsort

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get images from dataset path
imgs_path = "imgs"
imgs = natsort.natsorted([x for x in os.listdir(imgs_path)])

lists_file = "images_lists.txt"
files = os.listdir(imgs_path)
files = natsort.natsorted(files)
f = open(lists_file, "w+")
for x in files:
	print(x, file=f)
f.close()
files = [os.path.join(imgs_path, x) for x in files]
imgs_full = files


# Carregando o modelo
#dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

dinov2_vits14.to(device)

# Definindo o modelo no modo de avaliação (não atualiza os gradientes)
dinov2_vits14.eval()


extracted_features = []

# Ler imagens
for image_path in imgs_full:
    img = Image.open(image_path).convert('RGB')

    # Pré-processamento da imagem
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0).to(device)  # Adicionando a dimensão batch

    # Passando a imagem pelo modelo para extrair features
    with torch.no_grad():
        features = dinov2_vits14(img)

    print(image_path)

    extracted_features.append(features[0].tolist())

np.save("features_dino", extracted_features)
# As features extraídas estão no tensor 'features'
#print(features[0])
#print(features[0].shape)
