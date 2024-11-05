


# basing this model on this package with this link: https://github.com/timesler/facenet-pytorch
# basing on this model as well: https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch

# the goal being to adapt this to a raspberry pi system
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

help(MTCNN)



