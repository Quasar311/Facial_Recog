


# basing this model on this package with this link: https://github.com/timesler/facenet-pytorch
# basing on this model as well: https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch

# the goal being to adapt this to a raspberry pi system
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm


model = MTCNN()



from PIL import Image

path = '/Users/aidanbaydush/Code/Tinkering/face_rec/test_images/Aidan_1.jpeg'

img = Image.open(path)

width = img.width
height = img.height

mtcnn = MTCNN(image_size=___, margin=__)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path='/Users/aidanbaydush/Code/Tinkering/face_rec/modified images')

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))



