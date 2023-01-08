# main application file
from color_image import build_res_unet, MainModel, to_mau_anh_xam,to_mau_anh_xam_mo_hinh_1
import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
from PIL import Image, ImageMath
import matplotlib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from flask import Flask, render_template, request


app = Flask(__name__)

def predict_color_COCO(img_path):
    model_path = "./model_weights_80_epochs.pth"
    colorized_image = to_mau_anh_xam(model_path, img_path)
    image_colorized_output = "static/color/" + "colorized_ouput" + ".jpg"
    matplotlib.pyplot.imsave(image_colorized_output, colorized_image)
    return image_colorized_output

def predict_color_Landscape(img_path):
    model_path = "./final_model_weights_20_epochs.pth"
    colorized_image = to_mau_anh_xam(model_path, img_path)
    image_colorized_output = "static/color/" + "colorized_ouput_" + ".jpg"
    matplotlib.pyplot.imsave(image_colorized_output, colorized_image)
    return image_colorized_output

def predict_color_Landscape_mo_hinh_1(img_path):
    model_path = "./gen0_80.h5"
    colorized_image = to_mau_anh_xam_mo_hinh_1(model_path, img_path)
    image_colorized_output = "static/color/" + "colorized_ouput_" + ".jpg"
    plt.imsave(image_colorized_output, colorized_image)
    return image_colorized_output

def predict_color_COCO_mo_hinh_1(img_path):
    model_path = "./gen0_60.h5"
    colorized_image = to_mau_anh_xam_mo_hinh_1(model_path, img_path)
    image_colorized_output = "static/color/" + "colorized_ouput_" + ".jpg"
    plt.imsave(image_colorized_output, colorized_image)
    return image_colorized_output

def resize_imgae(img_path):
    img = PIL.Image.open(img_path)
    img_or = img.resize((256, 256))
    image_ground_truth = "static/ground_truth/" + "grounf_truth" + ".jpg"
    im1 = img_or.save(image_ground_truth)
    return image_ground_truth

# routes - Duong dan
@app.route("/", methods = ['GET', 'POST'])
def trang_chu():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return"Đồ án tốt nghiệp: GAN cho tô màu ảnh xám"

@app.route("/COCO", methods = ['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        img_path = resize_imgae(img_path)
        img_color_path = predict_color_COCO(img_path)
    return render_template("index.html",prediction = img_color_path, img_path = img_path, color_path= img_color_path)

@app.route("/landscape", methods = ['GET', 'POST'])
def get_hours_2():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        img_path = resize_imgae(img_path)
        img_color_path = predict_color_Landscape(img_path)
    return render_template("index.html",prediction = img_color_path, img_path = img_path, color_path= img_color_path)

@app.route("/landscapemohinh1", methods = ['GET', 'POST'])
def get_hours_3():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        img_path = resize_imgae(img_path)
        img_color_path = predict_color_Landscape_mo_hinh_1(img_path)
    return render_template("index.html",prediction = img_color_path, img_path = img_path, color_path= img_color_path)

@app.route("/COCOmohinh1", methods = ['GET', 'POST'])
def get_hours_4():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        img_path = resize_imgae(img_path)
        img_color_path = predict_color_COCO_mo_hinh_1(img_path)
    return render_template("index.html",prediction = img_color_path, img_path = img_path, color_path= img_color_path)

if __name__ == "__main__":
    app.run(debug= True)
        