import os
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import random
from pathlib import Path

##########데이터 로드

def aug(file_path, save_dir, size=None):

    image = Image.open(file_path)
    # image = image.convert('RGB') #'L': greyscale, '1': 이진화, 'RGB' , 'RGBA', 'CMYK'
    # if not size is None:
    #     image = image.resize((size, size))

    full_path = Path(file_path)
    file_name = full_path.name.split('.')[0]

    #밝기 10개
    for i in range(10) :
        p = (i + 1) * 0.1 + 0.8
        enhancer = ImageEnhance.Brightness(image)
        brightness_image = enhancer.enhance(p)
        brightness_image = brightness_image.resize((size, size))
        brightness_image.save(f'{save_dir}/{file_name}_{str(i)}.png') # brightness

    #회전 10개
    for i in range(10,20) :
        rotate_image = image.rotate(random.randint(-10, 10))
        rotate_image = rotate_image.resize((size, size))
        rotate_image.save(f'{save_dir}/{file_name}_{str(i)}.png') # rotate

    #좌우 대칭
    horizonal_flip_image = image.transpose(Image.FLIP_LEFT_RIGHT) 
    horizonal_flip_image = horizonal_flip_image.resize((size, size))
    horizonal_flip_image.save(f'{save_dir}/{file_name}_{str(20)}.png') # horizontal flip

    #상하 대칭
    vertical_flip_image = image.transpose(Image.FLIP_TOP_BOTTOM) 
    vertical_flip_image = vertical_flip_image.resize((size, size))
    vertical_flip_image.save(f'{save_dir}/{file_name}_{str(21)}.png') # vertical flip

    #확대 축소
    for i in range(22,27) :
        zoom = random.uniform(1.1, 1.5) #0.7 ~ 1.3
        width, height = image.size
        x = width / 2
        y = height / 2
        crop_image = image.crop((x - (width / 2 / zoom), y - (height / 2 / zoom), x + (width / 2 / zoom), y + (height / 2 / zoom)))
        zoom_image = crop_image.resize((width, height), Image.LANCZOS)
        zoom_image = zoom_image.resize((size, size))
        zoom_image.save(f'{save_dir}/{file_name}_{str(i)}.png')

    # #좌우 이동
    for i in range(27,30) :
        width, height = image.size
        shift = random.randint(0, round(width * 0.3))
        horizonal_shift_image = ImageChops.offset(image, shift, 0)
        horizonal_shift_image.paste((0), (0, 0, shift, height))
        horizonal_shift_image = horizonal_shift_image.resize((size, size))
        horizonal_shift_image.save(f'{save_dir}/{file_name}_{str(i)}.png') # horizontal shift

    # #상하 이동
    for i in range(30,33) :
        width, height = image.size
        shift = random.randint(0, round(height * 0.3))
        vertical_shift_image = ImageChops.offset(image, 0, shift)
        vertical_shift_image.paste((0), (0, 0, width, shift))
        vertical_shift_image = vertical_shift_image.resize((size, size))
        vertical_shift_image.save(f'{save_dir}/{file_name}_{str(i)}.png') # vertical shift

    # #기울기
    # #cx, cy = 0.1, 0
    # #cx, cy = 0, 0.1
    # cx, cy = 0, random.uniform(0.0, 0.3)
    # shear_image = image.transform(
    #     image.size,
    #     method=Image.AFFINE,
    #     data=[1, cx, 0,
    #         cy, 1, 0,])
    # shear_image.save('_6.png') # shear


def run(img_root, aug_dir):
    try :
        if not os.path.exists(aug_dir):
            os.makedirs(aug_dir)
    except OSError as e:
        print("error", e)

    root_path  = Path(img_root)
    print(">>> start augmentation")

    for obj in root_path.iterdir():
        print("augment img", obj.name)
        if obj.is_file() and obj.suffix == '.png':
            file_path = obj.absolute()
            aug(file_path, aug_dir, size=112)

    