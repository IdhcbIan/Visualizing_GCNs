"""
██╗███╗░░██╗███████╗███████╗██████╗░███████╗███╗░░██╗░█████╗░██╗░█████╗░██╗
██║████╗░██║██╔════╝██╔════╝██╔══██╗██╔════╝████╗░██║██╔══██╗██║██╔══██╗██║
██║██╔██╗██║█████╗░░█████╗░░██████╔╝█████╗░░██╔██╗██║██║░░╚═╝██║███████║██║
██║██║╚████║██╔══╝░░██╔══╝░░██╔══██╗██╔══╝░░██║╚████║██║░░██╗██║██╔══██║╚═╝
██║██║░╚███║██║░░░░░███████╗██║░░██║███████╗██║░╚███║╚█████╔╝██║██║░░██║██╗
╚═╝╚═╝░░╚══╝╚═╝░░░░░╚══════╝╚═╝░░╚═╝╚══════╝╚═╝░░╚══╝░╚════╝░╚═╝╚═╝░░╚═╝╚═╝

// Ian Bezerra - 2025 //

    Codigo de extracao de features de imagens!! Sera usado no servidor!!

"""


############// Selecionando os Modelos //#####################################


model_options = {
    1: "alexnet",
    2: "vit",
    3: "dinov2_vits14",
    4: "dinov2_vitb14", 
    5: "dinov2_vitl14",
    6: "dinov2_vitg14",
    7: "resnet18",
    8: "resnet34",
    9: "resnet50",
    10: "resnet101",
    11: "resnet152",
    12: "fbresnet152",
    13: "cafferesnet101",
    14: "inceptionresnetv2",
    15: "inceptionv4",
    16: "bninception",
    17: "resnext101_32x4d",
    18: "resnext101_62x4d",
    19: "dpn68",
    20: "dpn98",
    21: "dpn131",
    22: "dpn68b",
    23: "dpn92",
    24: "dpn107",
    25: "xception",
    26: "senet154",
    27: "se_resnet50",
    28: "se_resnet101",
    29: "se_resnet152",
    30: "se_resnext50_32x4d",
    31: "se_resnext101_32x4d",
    32: "pnasnet5large",
    33: "polynet",
    34: "densenet121",
    35: "densenet161",
    36: "densenet169",
    37: "densenet201",
    38: "squeezenet1_0",
    39: "squeezenet1_1",
    40: "vgg11",
    41: "vgg13",
    42: "vgg16",
    43: "vgg19",
    44: "vgg11_bn",
    45: "vgg13_bn",
    46: "vgg16_bn",
    47: "vgg19_bn",
    48: "ConvNext-large-224-22k-1k",
    49: "ConvNext-tiny-224",
    50: "swint",
    51: "dinov3-vits16-pretrain-lvd1689m",
    52: "dinov3-vitb16-pretrain-lvd1689m",
    53: "dinov3-vitl16-pretrain-lvd1689m"
}
print("Select an option:")


for number, option in model_options.items():
    print(f"{number}: {option}")

model_number = int(input("Enter model number: "))

if model_number in model_options:
    model_name = model_options[model_number]
    print(f"You selected: {model_name}")
else:
    print("Invalid selection.")


image_dir = "imgs"


############// Importando bibliotecas! //#####################################

import sys
import time
import os
import numpy as np
import natsort
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils
from ViT import vit_inference
from Convnext import aloi_convnext_inference
from SwinT import swint_inference

############// Configurando o modelo //#####################################


# Caso seja DinoV2 usar outra funcao
different_models = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14", "vit", "swint", "ConvNext-large-224-22k-1k", "ConvNext-tiny-224", "dinov3-vits16-pretrain-lvd1689m", "dinov3-vitb16-pretrain-lvd1689m", "dinov3-vitl16-pretrain-lvd1689m"]

if model_name not in different_models:
    # Configuração do modelo pré-treinado
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')  
    model.eval() 

    # Ajusta a última camada do modelo para obter as features antes da classificação final
    model.last_linear = pretrainedmodels.utils.Identity()


    #Check if we can use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)




############// Nova funcao para extrair features em batch!! //#####################################


def extract_features_batch(image_paths):
    # Carrega as transformações de imagem definidas para o modelo
    tf_img = utils.TransformImage(model)
    load_img = utils.LoadImage()

    # Process images in smaller batches
    #batch_size = len(image_paths) // 10  # Adjust based on your GPU memory
    batch_size = 128  # Adjust based on your GPU memory
    all_features = []
    
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
       
        # Processando imagens do batch atual
        input_tensors = []
        for img_path in batch_paths:
            input_img = load_img(img_path)
            input_tensor = tf_img(input_img)  # Ajusta a imagem para o tamanho esperado (ex.: 3x299x299)
            input_tensors.append(input_tensor)
        
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






















############// Selecionando as imagens!! //#####################################


# Diretório com as imagens
os.chdir(image_dir)
imgs_path = './'  # Caminho onde as imagens estão armazenadas
images = natsort.natsorted(os.listdir(imgs_path))  # Lista ordenada de imagens

# Inicialização da lista de features e arquivo de saída
#features = []  # Lista para armazenar as features extraídas
f = open("list.txt", "w+")  # Arquivo para registrar os nomes das imagens processadas
dataset_elements = []  # Lista auxiliar para os nomes das imagens





############// Preprocessando!! //#####################################


# Paths das imagens
image_paths = []

ini_p = time.time()
for i, img in enumerate(images):
    if ".jpg" not in img:  # Ignora arquivos que não sejam imagens .jpg
        continue

    # Salva o nome da imagem no arquivo de texto
    print(img, file=f)
    dataset_elements.append(img)

    # Define o caminho completo da imagem e adiciona à lista
    img_path = os.path.join(imgs_path, img)
    image_paths.append(img_path)

    # Log a cada 250 imagens processadas
    if i % 250 == 0:
        print(f"{i} images collected!")

f.close()  # Fecha o arquivo de texto
end_p = time.time()




############// Extraindo features!! //#####################################



ini = time.time()
print(f"Total images to process: {len(image_paths)}")

from DinoV2 import dinov2_inference
from DinoV3 import dinov3_inference

if model_name == "dinov2" or model_name == "dinov2_vits14" or model_name == "dinov2_vitb14" or model_name == "dinov2_vitl14" or model_name == "dinov2_vitg14":
    features = dinov2_inference(model_name, image_paths)    #Uma unica call para o funcao e com uma call para o modelo!!
elif model_name == "dinov3-vits16-pretrain-lvd1689m" or model_name == "dinov3-vitb16-pretrain-lvd1689m" or model_name == "dinov3-vitl16-pretrain-lvd1689m":
    features = dinov3_inference(model_name, image_paths)    #Uma unica call para o funcao e com uma call para o modelo!!
elif model_name == "vit":
    features = vit_inference(image_paths)
elif model_name == "ConvNext-large-224-22k-1k":
    features = aloi_convnext_inference(model_name, image_paths)
elif model_name == "ConvNext-tiny-224":
    features = aloi_convnext_inference(model_name, image_paths)
elif model_name == "swint":
    features = swint_inference(image_paths)
else:
    features = extract_features_batch(image_paths)    #Uma unica call para o funcao e com uma call para o modelo!!

end = time.time()
print(f"Time taken: {(end - ini) + (end_p - ini_p)} seconds" )


# Salvando as features
emb_path = "../Emb/" + model_name + "_emb.npy"

np.save(emb_path, features)
print("Done!")





############// Listas Ranqueadas!! //#####################################


import numpy as np
from sklearn.neighbors import BallTree


def run_ball_tree(features, k=100):
    """
    Constrói uma estrutura BallTree a partir das features e retorna os rankings dos vizinhos mais próximos.
    """


    # Verifica se as features são válidas
    if not isinstance(features, np.ndarray):
        raise ValueError("As 'features' devem ser um array do tipo numpy.ndarray.")
    if features.ndim != 2:
        raise ValueError("As 'features' devem ser um array 2D no formato (n_samples, n_features).")

    # Cria a estrutura BallTree
    tree = BallTree(features)

    # Realiza a consulta para encontrar os k vizinhos mais próximos
    _, rks = tree.query(features, k=k)

    return rks


rks = run_ball_tree(features)



############// Exportando as listas Ranqueadas!! //#####################################


import json

# Convert NumPy array to JSON and export
os.chdir("../Runs")
with open(model_name + "_output.json", "w") as json_file:
    # Format the JSON with proper indentation for readability
    json.dump(rks.tolist(), json_file, indent=4)
    # Ensure the file is properly closed and flushed
    json_file.flush()
    print(f"Data successfully exported to Runs/{model_name}_output.json")


############// Plotting the graph!! //#####################################
os.chdir("..")

rks_path = "./Runs/" + model_name + "_output.json"
with open(rks_path, "r") as f:
    print("FOUND FILE!!")
    rankings = json.load(f)

# Convert to numpy array
rankings = np.array(rankings)

os.chdir("./Plots")

import os
from Graph import plot_and_export

k_list = [10, 20, 30, 40, 60, 80]

plot_and_export(rankings, k_list, model_name)


#------------// End of the program //--------------------------
