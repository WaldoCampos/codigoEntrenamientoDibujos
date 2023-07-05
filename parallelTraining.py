from models.byol_parallel import BYOL_parallel
from models.sbir_byol import SBIR_BYOL
from models.byol_pytorch import BYOL
from data.custom_transforms import BatchTransform, ListToTensor, PadToSquare, SelectFromTuple
from data.pairs_dataset_transform import PairsDataset, pair_collate_fn

#from data.multi_pairs_dataset import MultiPairsDataset
from data.multi_pairs_dataset_transform import MultiPairsDataset
import numpy as np
import os

import sys
import torch
import torch.nn as nn

import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision import datasets
import warnings


# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.utils.data.distributed import DistributedSampler


epochs = 5


class TransFormSquare:

    def __init__(self, size):

        self.size = size
        self.x_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                PadToSquare(255),
                T.Resize((self.size,self.size)),
            ]
        )

        self.y_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                PadToSquare(255),
                T.Resize((self.size,self.size)),
            ]
        )

    def __call__(self, x, y):
        return self.x_transform(x) , self.y_transform(y)



ecommerce = {"images" : '/home/vision/smb-datasets/ecommerce/images/train/',
             "sketches" : '/home/vision/smb-datasets/ecommerce/ecommerce_pidinet/'}

sketchy = {"images" : '/home/vision/smb-datasets/SBIR/sketchy/256x256/photo/tx_000100000000/',
           "sketches" : '/home/vision/smb-datasets/SBIR/sketchy/256x256/sketch/tx_000100000000/'}

save_file = 'bimodal_byol_shoes_utils/checkpoints/Test/test_BYOl_parallel_training_epoch{}.pt'
pretrained_model = 'bimodal_byol_shoes_utils/checkpoints/bimodal_byol_resnet50_pretrained_sketchy_v5.pt'


def setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):

    #dataset = PairsDataset(
    #    ecommerce["images"],
    #    ecommerce["sketches"],
    #    transform = TransFormSquare(224)
    #)

    sketchy_dataset = MultiPairsDataset(
        sketchy["images"],
        sketchy["sketches"],
        separator="-",
        transform=TransFormSquare(224)
    )

    print("dataset ready")
    sampler = DistributedSampler(sketchy_dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)

    dataloader = torch.utils.data.DataLoader(
        sketchy_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler)

    return dataloader

def cleanup():
    dist.destroy_process_group()

encoder = models.resnet50(weights="IMAGENET1K_V2")

empty_transform = T.Compose([])


#learner = SBIR_BYOL(
#    encoder,
#    image_size=224,
#    hidden_layer="avgpool",
#    augment_fn = empty_transform,
#    cosine_ema_steps= epochs * epoch_size
#)



def main(rank, world_size):
    #setup the process groups
    setup(rank, world_size)

    #prepare the dataloader
    dataloader = prepare(rank, world_size, batch_size=100, num_workers=0)

    #instaDP

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model

    learner = SBIR_BYOL(
        encoder,
        image_size=224,
        hidden_layer="avgpool",
        augment_fn=empty_transform,
        cosine_ema_steps=epochs * len(dataloader)
    )

    torch.save(learner.state_dict(), save_file.format(epochs))
    learner.load_state_dict(torch.load( save_file.format(epochs)))

    learner.to(rank)

    model = DDP(learner, device_ids=[rank], output_device=rank, find_unused_parameters=True, gradient_as_bucket_view=True)
    

    optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)

    running_loss = np.array([], dtype=np.float32)
    for epoch in range(epochs):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)

        i = 0
        for step, (x, y) in enumerate(dataloader):

            optimizer.zero_grad(set_to_none=True)

            loss = model(x, y)#.to('cuda', dtype=torch.float)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            model.module.update_moving_average()

            running_loss = np.append(running_loss, [loss.mean().item()])
            sys.stdout.write('\rEpoch {}, batch {} - loss {:.4f}'.format(epoch+1, i+1, np.mean(running_loss)))

            i += 1
            #if i%(epoch_size/2)==0:
            #    torch.save(learner.state_dict(), save_file.format(epoch + 1))


        dist.barrier()

        if rank == 0:
            torch.save(model.module.state_dict(), save_file.format(epoch + 1))


        #torch.save(model.module.state_dict(), save_file.format(epoch + 1))
        running_loss = np.array([], dtype=np.float32)
        sys.stdout.write('\n')

    cleanup()

print("before main")

if __name__ == '__main__':

    #number of gpus
    world_size = 2

    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size
             )