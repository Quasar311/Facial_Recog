


# basing this model on this package with this link: https://github.com/timesler/facenet-pytorch
# basing on this model as well: https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch

# the goal being to adapt this to a raspberry pi system
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm


model = MTCNN()


mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


from PIL import Image

path = 'C:\\Users\\Aidan\\Code\\tinkering\\Face_rec\\face_rec\\'

img = Image.open(path)

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path=<optional save path>)

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))



