# File này sẽ chạy và tô ảnh
# python3.8
# pip install torchvision==0.10.1
# pip install fastai==2.4

import os
import glob                                       #|hỗ trợ việc tạo danh sách các tập tin từ việc tìm kiếm thư mục dùng ký tự thay thế
import time
import numpy as np
# from PIL import Image
from pathlib import Path
# from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2)
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or Đảm bảo rằng không sử dụng chuẩn hoá
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,| Khi cần tạo một số lớp lặp đi lặp lại
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, init='norm', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            # self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
            print("Please Load net_G ...")
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

import PIL
from PIL import Image, ImageMath
import numpy
import cv2
import matplotlib

# # from PIL import Image
# net_G = build_res_unet (n_input = 1, n_output = 2, size = 256)

# model = MainModel (net_G = net_G)
# # COCO
# # model.load_state_dict (torch.load ("D:/NguyenTranLongHao_18DT2/9_Do_An_Tot_Nghiep/Web_GAN_to_mau/final_model_weights.pth", map_location = device))
# model.load_state_dict (torch.load ("D:/NguyenTranLongHao_18DT2/9_Do_An_Tot_Nghiep/Web_GAN_to_mau/model_weights_80_epochs.pth", map_location = device))

# # LandScape
# # model.load_state_dict (torch.load ("D:/NguyenTranLongHao_18DT2/9_Do_An_Tot_Nghiep/Web_GAN_to_mau/final_model_weights_20_epochs_land.pth", map_location = device))
# # model.load_state_dict (torch.load ("D:/NguyenTranLongHao_18DT2/9_Do_An_Tot_Nghiep/Web_GAN_to_mau/final_model_weights_20_epochs.pth",map_location = device))

# image_test = "D:/NguyenTranLongHao_18DT2/9_Do_An_Tot_Nghiep/Web_GAN_to_mau/Coco_1_test" + ".jpg"
# # image_test = "D:/NguyenTranLongHao_18DT2/9_Do_An_Tot_Nghiep/Web_GAN_to_mau/Landscape_test" + ".jpg"
#   # image_test = "/content/drive/MyDrive/Đồ Án Tốt Nghiệp/COCO_Test/Show/Origin/show_" + str(i) + ".jpg"

# img = PIL.Image.open(image_test)
# img_or = img.resize((256, 256))
# gray_image = img_or.convert( 'L' )

# # to make it between -1 and 1
# img_trans = transforms.ToTensor()(img_or)[:1] * 2. - 1.
# # img_trans = ImageMath.eval("((a/255.0)*2.0)-1.0", a=img_lab)
# model.eval()
# with torch.no_grad():
#     preds = model.net_G(img_trans.unsqueeze(0))

# # Hiển thị ảnh đã tô màu
# plt.figure(figsize=(21,21))
# or_image = plt.subplot(1,3,1)
# or_image.set_title('Grayscale Input', fontsize=14)
# plt.imshow( gray_image , cmap='gray' )
# plt.axis("off")

# in_image = plt.subplot(1,3,2)
# in_image.set_title('Colorized Output', fontsize=14)
# colorized_image = lab_to_rgb(img_trans.unsqueeze(0), preds.cpu())[0]
# plt.imshow(colorized_image)
# # plt.axis("off")

# ou_image = plt.subplot(1,3,3)
# ou_image.set_title('Ground Truth', fontsize=14)
# # plt.imshow( img_or.resize((128, 128)))
# plt.imshow( img_or)
# # plt.axis("off")

# plt.show()



def to_mau_anh_xam(model_path, image_path):
    net_G = build_res_unet (n_input = 1, n_output = 2, size = 256)
    model = MainModel (net_G = net_G)
    model.load_state_dict (torch.load (model_path, map_location = device))
    
    img = PIL.Image.open(image_path)
    img_or = img.resize((256, 256))
    gray_image = img_or.convert( 'L' )

    # to make it between -1 and 1
    img_trans = transforms.ToTensor()(img_or)[:1] * 2. - 1.
    # img_trans = ImageMath.eval("((a/255.0)*2.0)-1.0", a=img_lab)
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img_trans.unsqueeze(0))

    # Hiển thị ảnh đã tô màu
    plt.figure(figsize=(21,21))

    plt.title('Colorized Output', fontsize=14)
    colorized_image = lab_to_rgb(img_trans.unsqueeze(0), preds.cpu())[0]
    # plt.imshow(colorized_image)
    # # plt.axis("off")
    # plt.show()
    # image_colorized_output = "/content/drive/MyDrive/Đồ Án Tốt Nghiệp/COCO_Test/Colorized_Output/" + "colorized_ouput_" + str(i) + ".jpg"
    # matplotlib.pyplot.imsave(image_colorized_output, colorized_image)

    return colorized_image

# a = to_mau_anh_xam("D:/NguyenTranLongHao_18DT2/9_Do_An_Tot_Nghiep/Web_GAN_to_mau/model_weights_80_epochs.pth", "D:/NguyenTranLongHao_18DT2/9_Do_An_Tot_Nghiep/Web_GAN_to_mau/Coco_1_test.jpg")
# plt.imshow(a)
# plt.show()

# Mo Hinh 1
# pip install tensorflow-addons
from tensorflow_addons.layers import SpectralNormalization
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.models import Model
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import img_to_array
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
tf.config.run_functions_eagerly(True)


def to_mau_anh_xam_mo_hinh_1(model_path, image_path):
    gen0 = tf.keras.models.load_model(model_path)

    color_test = image_path
    # gen0 = tf.keras.models.load_model('/content/drive/MyDrive/model3/gen0_'+str(h)+'.h5')
    img2 = cv2.cvtColor(cv2.imread(color_test), cv2.COLOR_BGR2RGB)
    new_img = cv2.cvtColor(cv2.imread(color_test), cv2.COLOR_BGR2GRAY)
    b=[]
    b.append(img_to_array(Image.fromarray(cv2.resize(img2,(128,128)))))
    b = np.array(b)
    b = (b/127.5) - 1
    a=[]
    a.append(img_to_array(Image.fromarray(cv2.resize(new_img,(128,128)))))
    a = np.array(a)
    a = (a/127.5) - 1
    gen_image = gen0(a , training = True)
    # fig( gen_image, b)
    image_co = (((gen_image[0] + 1.0) / 2.))
    image_co = np.asarray( image_co )
    image = (b[0] + 1.0) / 2.0
    image = np.asarray( image )
    return image_co

