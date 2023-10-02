import os
from models.SIMSIAM_model import SIMSIAM
from data.custom_transforms import PadToSquare
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
import torch
from torchvision.datasets import ImageFolder
import torchvision.models as models

import torchvision.transforms as T

from PIL import ImageFont, Image, ImageDraw

device = 'cuda:0'

quickdraw = {
    "queries": '/home/wcampos/datasets/quickdraw_3/test_known/',
    "catalogue": '/home/wcampos/datasets/quickdraw_3/test_known/',
    "red": '/home/wcampos/tests/codigoEntrenamientoDibujos/saved_models/sketch_only/resnet50/simsiam/10_epoch.pt',
}


Datos = quickdraw.copy()

def get_rank( idx, queries_embeddings, queries, catalogue_embeddings, catalogue):
    q_embeddings = queries_embeddings.numpy()
    c_embeddings = catalogue_embeddings.numpy()

    queries_names = queries.imgs
    catalogue_names = catalogue.imgs

    query_path = queries_names[idx][0]
    query_name = os.path.basename(query_path).split("_")[-3]

    distances = np.sqrt(np.sum(np.square(c_embeddings - q_embeddings[idx]), 1))
    indices = np.argsort(distances)

    r = 1
    rank = -1
    for j in indices:
        catalogue_path = catalogue_names[j][0]
        catalogue_name = os.path.basename(catalogue_path).split(".")[0]
        if query_name == catalogue_name:
            rank = 1/r
            break
        r = r + 1

    return rank


def get_catalogue_img_index(idx, queries, catalogue):

    queries_names = queries.imgs
    catalogue_names = catalogue.imgs

    query_path = queries_names[idx][0]
    query_name = os.path.basename(query_path).split("_")[-3]


    rel = -1
    for count, (catalogue_path, index) in enumerate(catalogue_names):
        catalogue_name = os.path.basename(catalogue_path).split(".")[0]

        if catalogue_name == query_name:
            rel = count
    return rel


def text_on_image(size=(224,224), texto="", tam = 90, pos=(30,60)):

    img = Image.new('RGB', size, color = (255, 255, 255))
    d = ImageDraw.Draw(img)

    font = ImageFont.load_default()
    d.text(pos, texto, font=font, fill=(0,0,0))

    return np.asarray(img)


def get_embeddings_labels(model, dataloader, mode):
    model.eval()
    embeddings = []
    labels = []
    for i, (batch, label) in enumerate(dataloader):
        batch = batch.to(device, dtype=torch.float)
        with torch.no_grad():
            current_projection, current_embedding = model(batch, return_embedding=mode)
        embeddings.append(current_embedding.to('cpu'))
        labels.append(label)
        sys.stdout.write('\rBatch {} done.'.format(i))
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0).numpy()


def frame_image(img, frame_width, value=(0,0,0)):
    b = frame_width
    ny, nx = img.shape[0], img.shape[1]
    framed_img = np.stack([
        np.ones((b+ny+b, b+nx+b))*value[0],
        np.ones((b+ny+b, b+nx+b))*value[1],
        np.ones((b+ny+b, b+nx+b))*value[2]
    ])
    framed_img = framed_img.transpose(1,2,0).astype(np.uint8)
    framed_img[b:-b, b:-b] = img
    return framed_img


def right_border_image(img, border_width, value=(0,0,0)):
    b = border_width
    ny, nx = img.shape[0], img.shape[1]
    framed_img = np.stack([
        np.ones((ny, nx+b))*value[0],
        np.ones((ny, nx+b))*value[1],
        np.ones((ny, nx+b))*value[2]
    ])
    framed_img = framed_img.transpose(1,2,0).astype(np.uint8)
    framed_img[:,:-b] = img
    return framed_img


def plot_best_10_multiple(queries, queries_embeddings, queries_labels, catalogue, catalogue_embeddings, catalogue_labels, selected_indices, discard_first=False, lines_width=8):

    q_embeddings = queries_embeddings.numpy()
    c_embeddings = catalogue_embeddings.numpy()
    
    fig, axes = plt.subplots(nrows=len(selected_indices) + 1, ncols=13, figsize=(int(13*1.5), int(len(selected_indices)*1.5)))

    img_text = text_on_image(texto="original", tam=56)

    axes[0][0].imshow(img_text.astype(np.uint8))
    axes[0][0].axis('off')

    img_text = text_on_image(texto="dibujo", tam=56)

    axes[0][1].imshow(img_text.astype(np.uint8))
    axes[0][1].axis('off')

    img_text = text_on_image(texto="rango", tam=56)

    img_text = right_border_image( img_text.astype(np.uint8),lines_width)

    axes[0][2].imshow(img_text.astype(np.uint8))
    axes[0][2].axis('off')

    for k in range(3, 13):
        img_text = text_on_image(texto="", tam=55)

        axes[0][k].imshow(img_text.astype(np.uint8))
        axes[0][k].axis('off')

    for i, selected_idx in enumerate(selected_indices):
        i = i+1

        distances = np.sqrt(np.sum(np.square(c_embeddings - q_embeddings[selected_idx]), 1))
        indices = np.argsort(distances)

        if discard_first:
            best_indices = indices[1:11]
        else:
            best_indices = indices[0:10]
        best = []
        best_labels = []
        for idx in best_indices:
            best.append(catalogue[idx])
            best_labels.append(catalogue_labels[idx])

        selected = queries[selected_idx][0].numpy().transpose(1,2,0)
        selected_label = queries_labels[selected_idx]

        axes[i][1].imshow(selected.reshape(224,224, 3).astype(np.uint8))
        axes[i][1].axis('off')

        for j, (image, label) in enumerate(best):
            image = image.numpy().transpose(1,2,0)
            axes[i][j+3].imshow(image.reshape(224,224,3).astype(np.uint8))
            axes[i][j+3].axis('off')
    return fig, axes


def compute_map(sim_matrix, q_labels, c_labels, k=5):
    sorted_pos = np.argsort(-sim_matrix, axis = 1)                
    AP = []
    sorted_pos_limited = sorted_pos[:, 1:] if k == -1 else sorted_pos[:, 1:k + 1]
    for i in np.arange(sorted_pos_limited.shape[0]) :
        ranking = c_labels[sorted_pos_limited[i,:]]                 
        pos_query = np.where(ranking == q_labels[i])[0]
        pos_query = pos_query + 1 
        if len(pos_query) == 0 :
            AP_q = 0
        else :
            recall = np.arange(1, len(pos_query) + 1)
            pr = recall / pos_query
            AP_q = np.mean(pr)
        AP.append(AP_q)
        #print('{} -> mAP = {}'.format(len(pos_query), AP_q))
                        
    mAP = np.mean(np.array(AP))        
    return mAP

#Transformaciones para las imágenes
transform_queries = T.Compose([
    PadToSquare(fill=255),
    T.Resize((224,224)),
    T.PILToTensor()
])

transform_catalogue = T.Compose([
    PadToSquare(fill=255),
    T.Resize((224,224)),
    T.PILToTensor()
])

#Cargar archivos
queries = ImageFolder(
    root = Datos["queries"],
    transform = transform_queries)

catalogue = ImageFolder(
    root = Datos["catalogue"],
    transform = transform_catalogue)

queries_loader = torch.utils.data.DataLoader(queries, batch_size=512, shuffle=False)
catalogue_loader = torch.utils.data.DataLoader(catalogue, batch_size=512, shuffle=False)

encoder = models.resnet50()

learner = SIMSIAM(
    encoder,
    image_size=224,
    hidden_layer='avgpool',
)

# red color
learner.load_state_dict(torch.load(Datos["red"], map_location=torch.device(device)), strict=False)
learner = learner.to(device)

queries_embeddings, queries_labels = get_embeddings_labels(learner, queries_loader, True)
catalogue_embeddings, catalogue_labels = get_embeddings_labels(learner, catalogue_loader, True)

queries_embeddings = queries_embeddings / np.linalg.norm(queries_embeddings, ord=2, axis=1, keepdims=True)

catalogue_embeddings = catalogue_embeddings / np.linalg.norm(catalogue_embeddings, ord=2, axis=1, keepdims=True)

similarity_matrix = np.matmul(queries_embeddings, catalogue_embeddings.T)

final_metric = compute_map(similarity_matrix, queries_labels, catalogue_labels, k=5)

print(f"mAP@5 del modelo: {final_metric}")
