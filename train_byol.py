# from models.BYOL2_model import BYOL2
from models.BYOL import BYOL2
from data.custom_transforms import BatchTransform, ListToTensor, PadToSquare, SelectFromTuple, RandomLineSkip, RandomRotation
from data.pairs_dataset import PairsDataset, pair_collate_fn
from data.custom_transforms import RandomLiquify, RandomMovement, HueAdjust, SaturationAdjust

from data.multi_pairs_dataset import MultiPairsDataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import torch
from torch.utils.data import Subset
import torchvision.models as models
import torchvision.transforms as T
import time
import datetime
import warnings
import os

# from torchinfo import summary


quickdraw = {"images": "/home/wcampos/datasets/quickdraw_3/train/",
             "sketches": "/home/wcampos/datasets/quickdraw_3/train/"}

save_file = '/home/wcampos/tests/codigoEntrenamientoDibujos/saved_models/sketch_only/resnet50/byol/{}_epoch.pt'
# pretrained_model = '/home/wcampos/tests/codigoEntrenamientoDibujos/saved_models/sketch_only/resnet50/byol/{}_epoch.pt'
if not os.path.exists(os.path.split(save_file)[0]):
    os.makedirs(os.path.split(save_file)[0])

# device = "cuda:0"
device = 'cpu'
batch_size = 64
torch.cuda.empty_cache()

quickdraw_dataset = PairsDataset(
    quickdraw["images"],
    quickdraw["sketches"]
)

# # resize the images for the net
# transforms_1 = T.Compose([
#     BatchTransform(SelectFromTuple(0)),
#     BatchTransform(PadToSquare(255)),
#     BatchTransform(T.Resize((224, 224))),
#     BatchTransform(T.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1))),
#     ListToTensor(device, torch.float),
# ])
# transforms_2 = T.Compose([
#     BatchTransform(SelectFromTuple(1)),  # sketch
#     BatchTransform(PadToSquare(255)),
#     BatchTransform(T.Resize((224, 224))),
#     BatchTransform(T.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1))),
#     ListToTensor(device, torch.float),
# ])

sketch_transform_1 = T.Compose([
    BatchTransform(SelectFromTuple(0)),
    BatchTransform(T.Resize((224, 224))),
    BatchTransform(RandomLineSkip(prob=0.5, skip=0.1)),
    BatchTransform(RandomRotation(prob=0.5, angle=30)),
    BatchTransform(T.RandomHorizontalFlip(p=0.5)),
    BatchTransform(T.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1))),
    ListToTensor(device, torch.float),
])

sketch_transform_2 = T.Compose([
    BatchTransform(SelectFromTuple(1)),
    BatchTransform(T.Resize((224, 224))),
    BatchTransform(RandomLineSkip(prob=0.5, skip=0.1)),
    BatchTransform(RandomRotation(prob=0.5, angle=30)),
    BatchTransform(T.RandomHorizontalFlip(p=0.5)),
    BatchTransform(T.RandomResizedCrop(224, scale=(0.8, 1), ratio=(1, 1))),
    ListToTensor(device, torch.float),
])

# BatchTransform(RandomLiquify(prob=0.9, size=(224,224))),
# BatchTransform(RandomMovement(0.5)),

train_loader = torch.utils.data.DataLoader(
    quickdraw_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=pair_collate_fn,
    num_workers=4
)

encoder = models.resnet50(weights="IMAGENET1K_V2")
empty_transform = T.Compose([])
epochs = 10
epoch_size = len(train_loader)

learner = BYOL2(
    encoder,
    image_size=224,
    hidden_layer="avgpool",
    augment_fn=empty_transform,
    cosine_ema_steps=epochs*epoch_size,
)

# summary(learner, input_size=(64, 3, 224, 224))

# se agregan las transformaciones a la red
learner.augment1 = sketch_transform_1
learner.augment2 = sketch_transform_2

# optimizador
opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

# learner.load_state_dict(torch.load(pretrained_model))
# torch.save(learner.state_dict(), save_file.format(epochs))
# learner.load_state_dict(torch.load( save_file.format(epochs)))

learner = learner.to(device)
learner.train()
# filehandler = open('weightFiles/filehandler.txt', 'w')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    running_loss = np.array([], dtype=np.float32)
    for epoch in range(epochs):
        i = 0
        # for images in Loaders[int(epoch/epochsForLoader)]:
        t0 = time.time()
        for images in train_loader:
            loss = learner(images)  # .to('cuda', dtype=torch.float)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
            running_loss = np.append(running_loss, [loss.item()])
            elapsed_time = time.time() - t0
            elapsed_timedelta = datetime.timedelta(seconds=elapsed_time)
            elapsed_timedelta = elapsed_timedelta - \
                datetime.timedelta(microseconds=elapsed_timedelta.microseconds)
            elapsed_time_formatted = str(elapsed_timedelta)
            sys.stdout.write(
                '\rEpoch {}, batch {} - loss {:.4f} - elapsed time {}'.format(epoch+1, i+1, np.mean(running_loss), elapsed_time_formatted))
            # filehandler.write('Epoch {}, batch {} - loss {:.4f}\n'.format(epoch+1, i+1, np.mean(running_loss)))
            # filehandler.flush()
            i += 1
            # if i%(epoch_size/2)==0:
            #    torch.save(learner.state_dict(), save_file.format(epochs))
        final_time = time.time() - t0
        final_timedelta = datetime.timedelta(seconds=final_time)
        final_timedelta = final_timedelta - \
            datetime.timedelta(microseconds=final_timedelta.microseconds)
        final_time_formatted = str(final_timedelta)
        print(f"\nTiempo en epoch {epoch}: {final_time_formatted}")
        # if epoch % 2 == 0:
        torch.save(learner.state_dict(), save_file.format(epoch + 1))
        running_loss = np.array([], dtype=np.float32)
        sys.stdout.write('\n')
# filehandler.close()
