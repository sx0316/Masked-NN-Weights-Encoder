"""
Download a language model and load its weights layer by layer as matrices
1. Download albert model
2. Load the model
3. Load the weights
4. Load the weights layer by layer as matrices
5. Save the matrices

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.animation as animation
import sys
import os
import time
import math
import random
import pickle
import copy
import argparse
import collections
import csv
import json
import re
import string
import unicodedata
import itertools
import scipy
import scipy.stats
import scipy.io
import scipy.misc
import scipy.cluster.hierarchy as sch
import scipy.cluster.vq as vq
import scipy.linalg as la
import scipy.spatial.distance as sd
import scipy.ndimage.filters as fi
import scipy.ndimage.morphology as mo
import scipy.ndimage.measurements as me
import scipy.ndimage.interpolation as inp
import scipy.ndimage.filters as fl
import scipy.ndimage.morphology as mr
import scipy.ndimage.measurements as ms
import scipy.ndimage.interpolation as ip
import scipy.ndimage.filters as f
import scipy.ndimage.morphology as m
import scipy.ndimage.measurements as mn
import scipy.ndimage.interpolation as i
import scipy.ndimage.filters as f
import scipy.ndimage.morphology as m
import scipy.ndimage.measurements as mn
import scipy.ndimage.interpolation as i
import scipy.ndimage.filters as f
import scipy.ndimage.morphology as m
import scipy.ndimage.measurements


# 1. Download albert model
model = models.albert.albert.from_pretrained('albert-base-v2')
# 2. Load the model
model.load_state_dict(torch.load('albert-base-v2-pytorch_model.bin'))
# 3. Load the weights
model.load_state_dict(torch.load('albert-base-v2-pytorch_model.bin'))
# 4. Load the weights layer by layer as matrices
matrices = []
for name, param in model.named_parameters():
    if 'weight' in name:
        matrices.append(param.detach().numpy())
# 5. Save the matrices
with open('albert-base-v2-pytorch_model.bin', 'wb') as f:
    pickle.dump(matrices, f)
# 6. Load the matrices