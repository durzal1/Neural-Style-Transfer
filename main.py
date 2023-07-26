import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import *
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image

# Hyperparameters
LEARNING_RATE = .01
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 200
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "final.pth.tar"
SOURCE_PATH = "images/source"
STYLE_PATH = "images/style"
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
ind1 = 5
ind2 = 4

def save_checkpoint(state, filename="final.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])




transform = A.Compose(
    [
        A.Resize(width=IMAGE_HEIGHT, height=IMAGE_WIDTH),
        # A.RandomCrop(width=1280, height=720),
        # A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9), # don't think i need this tbh
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
def load_and_preprocess_images(source_path, style_path, ind, ind2):

    source_path = f'{source_path}/source_{ind}.png'
    style_path = f'{style_path}/style_{ind2}.jpg'

    content_img = np.array(Image.open(source_path).convert('RGB'))
    style_img = np.array(Image.open(style_path).convert('RGB'))

    augmentations = transform(image=content_img)
    augmentations2 = transform(image=style_img)
    source_path_img = augmentations["image"]
    style_img = augmentations2["image"]

    return source_path_img, style_img




# vgg model that we will be using the features of
class vgg(nn.Module):

    def __init__(self):
        super(vgg, self).__init__()

        # 0: block1_conv1
        # 5: block2_conv1
        # 10: block3_conv1
        # 19: block4_conv1
        # 28: block5_conv1
        self.req_features = ['0', '5', '10', '19', '28']

        self.model = torchvision.models.vgg19(pretrained=True).features


    def forward(self,x):
        features = []

        for name, layer in enumerate(self.model):

            x = layer(x)

            if (str(name) in self.req_features):
                features.append(x)

        return features

# loss function
def calcLossStyle(feature_style, feature_gen):

    num_channels, height, width = feature_style.size()

    feature = feature_style.view(num_channels, height*width)

    gen = feature_gen.view(num_channels, height*width)
    it = feature.t()

    # style loss
    gram_matrix_style = torch.mm(feature, feature.t())
    gram_matrix_gen = torch.mm(gen, gen.t())

    style_loss = nn.functional.mse_loss(gram_matrix_style, gram_matrix_gen)

    return style_loss

def calcLossSource(feature_source, feature_gen):
    # source loss
    source_loss = nn.functional.mse_loss(feature_source, feature_gen)

    return source_loss

def loss_fn(features_style,features_gen,features_source):

    style_loss = 0
    source_loss = 0

    style_weight = .3
    source_weight = 1

    for style, generated, source, in zip(features_style, features_gen, features_source):

        style_loss += calcLossStyle(style, generated)
        source_loss += calcLossSource(source, generated)

    return style_weight * style_loss + source_weight * source_loss


# source and style image we will be using
source, style = load_and_preprocess_images(SOURCE_PATH, STYLE_PATH,ind1,ind2)

model = vgg().to(DEVICE)

source = source.cuda()
style = style.cuda()

# create the random image
gen = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH,3)
augmentations = transform(image=gen)
gen = augmentations["image"]

gen = gen.cuda()
gen = source.clone()
gen = gen.requires_grad_(True)


optimizer=optim.Adam([gen],lr=LEARNING_RATE)




# training

for epoch in range(EPOCHS):

    mean_loss = []

    features_style = model(style)
    features_gen = model(gen)
    features_source = model(source)

    loss = loss_fn(features_style,features_gen,features_source)

    mean_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"epoch: {epoch}")
    print (f"loss is {loss.item()}")

torchvision.utils.save_image(gen, "images/results/obama3.png")


gen = gen.requires_grad_(False)

# torchvision.utils.save_image()
# fig, ax = plt.subplots(3, figsize=(5, 10))
# ax[0].imshow(source.cpu().permute(1, 2, 0).contiguous())
# ax[1].imshow(style.cpu().permute(1, 2, 0).contiguous())
# ax[2].imshow(gen.cpu().permute(1, 2, 0).contiguous())
# plt.savefig(f"images/results/done1.png")

