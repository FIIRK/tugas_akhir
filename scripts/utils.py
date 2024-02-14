from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchvision.transforms.functional as F_
import random

def augment(images):
    opr = random.choice(('RESIZED_CROP', 'ROTATE', 'BLUR', 'NONE'))
    image = images[0]

    if opr == 'RESIZED_CROP':
        r = 0.6 + 0.4 * random.random()
        limx, limy = int(image.size[0]*r), int(image.size[1]*r)
        x = random.randint(0, image.size[0]-limx)
        y = random.randint(0, image.size[1]-limy)
        images = [F_.resized_crop(i, x, y, limx, limy, (1024, 1024)) for i in images]

    elif opr == 'ROTATE':
        angle = 360 * random.random()
        images = [F_.resize(F_.rotate(i, angle, expand=True), (1024, 1024)) for i in images]

    elif opr == 'BLUR':
        images = [F_.gaussian_blur(i, random.choice([3, 5])) for i in images]

    return images

def get_result(inputs, model):
  Tinput = inputs.unsqueeze(dim=0)

  with torch.no_grad():
    output = model(Tinput)

  output = torch.sigmoid(output)
  return output.cpu().squeeze()

def arr_to_img(arr, threshold=0, cs={'R':(1, 1), 'G':(0, 1), 'B':(None, 1)}):
  arrt = np.ceil(arr if not threshold else arr * (arr >= threshold))
  if len(arrt.shape) == 3:
    temp = np.zeros_like(arrt)
    for i in range(arrt.shape[1]):
      for j in range(arrt.shape[2]):
        max_index = np.argmax(arrt[:, i, j])
        temp[max_index, i, j] = arr[max_index, i, j]
    R = np.zeros_like(temp[0]) if cs['R'][0] is None else (temp[cs['R'][0]] * cs['R'][1])
    G = np.zeros_like(temp[0]) if cs['G'][0] is None else (temp[cs['G'][0]] * cs['G'][1])
    B = np.zeros_like(temp[0]) if cs['B'][0] is None else (temp[cs['B'][0]] * cs['B'][1])
    arrt = np.stack([R, G, B], axis=2)
  img = Image.fromarray(np.uint8(arrt * 255))
  return img
