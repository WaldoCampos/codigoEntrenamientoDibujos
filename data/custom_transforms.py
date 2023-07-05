import cairocffi as cairo
import cairosvg
from io import BytesIO
import math
import numpy as np
from PIL import Image
import random
import skimage.feature
from skimage import transform

import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as T

class BatchTransform(object):
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image_batch):
        transformed_image_batch = []
        for image in image_batch:
            transformed_image_batch.append(self.transform(image))
        return transformed_image_batch


class ListToTensor(object):
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
    
    def __call__(self, list_batch):
        return torch.stack(list_batch, axis=0).to(self.device, dtype=self.dtype)

class PadToSquare(object):
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            _, height, width = image.shape
        else:
            width, height = image.size
        l_pad, t_pad, r_pad, b_pad = 0, 0, 0, 0
        if height < width:
            t_pad = int((width-height)/2)
            b_pad = (width-height)-t_pad
        elif width < height:
            l_pad = int((height-width)/2)
            r_pad = (height-width)-l_pad
        image = T.functional.pad(image, (l_pad, t_pad, r_pad, b_pad), fill=self.fill)
        return image


#class PadToSquare(object):
#    def __init__(self, fill=0):
#        self.fill = fill
#
#    def __call__(self, image):
#
#        _, height, width = image.shape
#
#        l_pad, t_pad, r_pad, b_pad = 0, 0, 0, 0
#        if height < width:
#            t_pad = int((width-height)/2)
#            b_pad = (width-height)-t_pad
#        elif width < height:
#            l_pad = int((height-width)/2)
#            r_pad = (height-width)-l_pad
#        image = T.functional.pad(image, (l_pad, t_pad, r_pad, b_pad), fill=self.fill)
#        return image





class SelectFromTuple(object):
    def __init__(self, index):
        self.index = index
    
    def __call__(self, data_tuple):
        return data_tuple[self.index]


class GaussianBlur(object):
    def __init__(self, kernel=5, sigma=(0.1, 2.0)):
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, image):
        blurrer = T.GaussianBlur(self.kernel, self.sigma)

        return blurrer(image)


class RandomMovement(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        p = random.uniform(0, 1)
        
        if p < self.prob:
            aff = T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), fill=255)
            image = aff(image)

        return image



def liquify(img,
            size = (400, 400),  # tamaño de la imagen
            curve = 9,
            strength = 0.15,
            steps = 2           # cantidad de centros que se van a crear
            ):

    # campo vectorial
    fieldflow = torch.zeros((size[0], size[1],2))

    init_x, init_y = torch.meshgrid(torch.linspace(-1, 1, size[0]),
                                    torch.linspace(-1, 1, size[1]),
                                    indexing='ij')

    result = img

    for i in range(steps):

        # selección del centro
        center = (random.uniform(-0.75, 0.75), random.uniform(-0.75, 0.75))

        # calculo de las coordenadas polares
        rho = torch.sqrt((init_x - center[0]) ** 2 + (init_y - center[1]) ** 2)
        theta = torch.arctan2(init_y - center[1], init_x - center[0])

        # liquify, se acercan los pixeles hacia el centro seleccionado, dependiendo de curve y strength
        # curve:     define el radio de los pixeles afectados
        # strength:  define cuanto van a moverse los pixeles
        auxrho = 1 * (torch.exp(-(rho*curve) + np.log(strength)))



        # transformacion de coordenadas polares a cartesianas
        fieldflow[:,:,0] = init_x + auxrho * torch.cos(theta)
        fieldflow[:,:,1] = init_y + auxrho * torch.sin(theta)

        fieldflow = fieldflow.unsqueeze(dim=0)

        # aplicación del campo vectorial a la imagen
        result = torch.nn.functional.grid_sample(result.float(), fieldflow.float(), padding_mode='border', align_corners=True)

        fieldflow = fieldflow.squeeze(dim=0)

    return result.int()

class RandomLiquify(object):
    def __init__(self, prob=0.5, size=(400,400), curve=4, strength=0.05, steps=6):
        self.prob = prob
        self.size = size
        self.curve = curve
        self.strength = strength
        self.steps = steps

    def __call__(self, image):

        p = random.uniform(0, 1)

        if p < self.prob:
            image = image.unsqueeze(dim=0)
            image = liquify(image, self.size, self.curve, self.strength, self.steps)
            image = image.squeeze(dim=0)

        return image

class HueAdjust(object):
    def __init__(self, prob=0.5, hue_shift=0.06):
        self.prob = prob
        self.hue_shift = hue_shift

    def __call__(self, image):

        p = random.uniform(0, 1)
        t = random.uniform(0, 1)

        if p < self.prob:
            if t > 0.5:
                image = F.adjust_hue(image, self.hue_shift)
            elif t < 0.5:
                image = F.adjust_hue(image, -self.hue_shift)

        return image


class SaturationAdjust(object):
    def __init__(self, prob=0.5, saturation_shift=0.06):
        self.prob = prob
        self.saturation_shift = saturation_shift

    def __call__(self, image):

        p = random.uniform(0, 1)
        t = random.uniform(0, 1)

        if p < self.prob:
            if t > 0.5:
                image = F.adjust_saturation(image, 1 + self.saturation_shift)
            elif t < 0.5:
                image = F.adjust_saturation(image, 1 - self.saturation_shift)

        return image