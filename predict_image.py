from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.applications.mobilenet import relu6, DepthwiseConv2D
import keras.backend as K
import boto3
#import mritopng
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import png
import pydicom
import os, shutil

def save_img(test, img_dir, image, im, box, pic):
    region = im.crop(box)
    img_name = os.path.basename(image)
    if np.average(np.array(region.getdata())) > 4:
        region.save('{}_{}_{}_{}.png'.format(test+img_name[:-4], str(pic).zfill(5), str(int(box[0])).zfill(5), str(int(box[1])).zfill(5)))

def mri_to_png(mri_file_path, png_file_path):
    mri_file = open(mri_file_path, 'rb')
    png_file = open(png_file_path, 'wb')

    # Extracting data from the mri file
    plan = pydicom.read_file(mri_file)
    shape = plan.pixel_array.shape

    #Convert to float to avoid overflow or underflow losses.
    image_2d = plan.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    
    #Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Writing the PNG file
    w = png.Writer(shape[1], shape[0], greyscale=True)
    w.write(png_file, image_2d_scaled)

    png_file.close()
    
def predict_image(image):
    if image.endswith('.dcm'):
        if not os.path.isfile(image[:-4]+'.png'):
            mri_to_png(image, image[:-4]+'.png')
            print('converted dcm to png')
        image = image[:-4]+'.png'

    elif not image.endswith('.jpg') and not image.endswith('.png'):
        print('image must be .dcm, .png, or .jpg format')
    
    cv_img = cv2.imread(image,0)
    print('loaded image')
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(cv_img)
    cv2.imwrite(image[:-4]+'_clahe.png',img_clahe)
    
    print('applied CLAHE to image')
    
    img_dir = image[:-4]+'/'
    test = img_dir+'test/'
    
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)
    
    os.mkdir(img_dir)
    os.mkdir(test)
    
    print('directory structure made')
    
    im = Image.open(image[:-4]+'_clahe.png')
    
    slice_size=224
    offset=.5
    width, height = im.size
    pic = 1 
    window_slide = slice_size * (1 - offset)
    
    for i in np.arange(0, width - slice_size, window_slide):
        for j in np.arange(0, height - slice_size, window_slide):
            box = (i, j, i + slice_size, j + slice_size)
            save_img(test, img_dir, image, im, box, pic)
            pic += 1
        box = (i, height - slice_size, i + slice_size, height)
        save_img(test, img_dir, image, im, box, pic)
        pic += 1
    box = (width - slice_size, height - slice_size, width, height)
    save_img(test, img_dir, image, im, box, pic)
    pic += 1
    
    print('images cut up')
    
    K.clear_session()
    with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
        model = load_model('1_epoch.h5')
        
    print('model loaded')

    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(img_dir,
                                                                            target_size=(224, 224),
                                                                            batch_size=10,
                                                                            class_mode=None)
    
    pred = model.predict_generator(test_generator,verbose=0)
    
    print('made predictions')
    
    predictions = np.fmax(np.zeros(pred.shape),((pred-.5)*2)-.2)[:,0]
    
    imgs = sorted(os.listdir(img_dir+'test'))
    
    base = Image.open(image[:-4]+'_clahe.png')
    
    rgb = np.array(base.convert('RGB').getdata()).reshape(base.size[1],base.size[0],3)
    predictions = (predictions*100).astype(np.int8)
    
    for i,img in enumerate(imgs):
        y = int(img[-9:-4])
        x = int(img[-15:-10])

        rgb[y:y+slice_size,x:x+slice_size,0]+=predictions[i]

    final = Image.fromarray(np.uint8(rgb))
    print('final image made')
    
    path = image[:-4]+'_final.png'
    os.remove(image)
    shutil.rmtree(image[:-4])
    final.save(path)
    final.save('/var/www/html/flaskapp/'+path)    
    #return final, np.max(pred[:,0]*100)
    return path
