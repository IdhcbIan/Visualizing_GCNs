# This is the main file for the server application!!


"""

(Images, Model) -> System -> Rankings(Json)

"""

model_options = {1: "alexnet", 2: "resnet50", 3: "resnet152"}
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
#--------------------------------------

# Importação de bibliotecas necessárias
import sys
import time
import os
import numpy as np
import natsort
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils

def extract_features(path_img):
    """
    Função para extrair features de uma imagem usando um modelo pré-treinado.

    Args:
        path_img (str): Caminho para a imagem.

    Returns:
        features (list): Lista de features extraídas da imagem.
    """
    # Carrega as transformações de imagem definidas para o modelo
    tf_img = utils.TransformImage(model)

    # Carrega a imagem usando a função utilitária de carregamento
    load_img = utils.LoadImage()
    input_img = load_img(path_img)

    # Aplica as transformações na imagem
    input_tensor = tf_img(input_img)  # Ajusta a imagem para o tamanho esperado (ex.: 3x299x299)
    input_tensor = input_tensor.unsqueeze(0)  # Adiciona dimensão extra para batch (1x3x299x299)

    # Converte para tensor PyTorch sem gradiente
    input = torch.autograd.Variable(input_tensor, requires_grad=False)

    # Extrai as features usando o modelo
    features = model(input)  # Extração de features
    features = features.data.cpu().numpy().tolist()[0]  # Converte para uma lista no formato NumPy
    return features

# Configuração do modelo pré-treinado
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')  
model.eval() 

# Ajusta a última camada do modelo para obter as features antes da classificação final
model.last_linear = pretrainedmodels.utils.Identity()

# Diretório com as imagens
os.chdir(image_dir)
imgs_path = './'  # Caminho onde as imagens estão armazenadas
images = natsort.natsorted(os.listdir(imgs_path))  # Lista ordenada de imagens

# Inicialização da lista de features e arquivo de saída
features = []  # Lista para armazenar as features extraídas
print(model_name)  # Exibe o nome do modelo
f = open("list.txt", "w+")  # Arquivo para registrar os nomes das imagens processadas
dataset_elements = []  # Lista auxiliar para os nomes das imagens

ini = time.time()
# Processamento das imagens
for i, img in enumerate(images):
    if ".jpg" not in img:  # Ignora arquivos que não sejam imagens .jpg
        continue

    # Salva o nome da imagem no arquivo de texto
    print(img, file=f)
    dataset_elements.append(img)

    # Define o caminho completo da imagem
    img = os.path.join(imgs_path, img)

    # Log a cada 250 imagens processadas
    if i % 250 == 0:
        print(f"{i} images processed!")

    # Extrai features da imagem e adiciona à lista
    features.append(extract_features(img))

f.close()  # Fecha o arquivo de texto

end = time.time()
print(f"Time taken: {end - ini} seconds")
# Salva as features extraídas em um arquivo .npy
features = np.array(features)  # Converte para array NumPy
#np.save("features", features)  # Salva as features no disco
print("Done!")




#------------// Listas Ranqueadas //--------------------------


from sklearn.neighbors import BallTree
import numpy as np

def run_ball_tree(features, k=100):
    """
    Constrói uma estrutura BallTree a partir das features e retorna os rankings dos vizinhos mais próximos.

    Args:
        features (numpy.ndarray): Array de características (features) usado para construir a árvore.
                                  Formato esperado: (n_samples, n_features).
        k (int, opcional): Número de vizinhos mais próximos a serem retornados para cada amostra. Default: 100.

    Returns:
        - rks (numpy.ndarray): Índices dos k vizinhos mais próximos para cada ponto no array de entrada.
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




#------------// Sending the output out //--------------------------


import json
os.chdir("../Runs")

# Convert NumPy array to JSON and export
with open(model_name + "_simple_output.json", "w") as json_file:
    # Format the JSON with proper indentation for readability
    json.dump(rks.tolist(), json_file, indent=4)
    # Ensure the file is properly closed and flushed
    json_file.flush()
    print(f"Data successfully exported to Runs/{model_name}_simple_output.json")


#------------// End of the program //--------------------------