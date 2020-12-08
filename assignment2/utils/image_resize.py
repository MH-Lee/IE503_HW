from PIL import Image
from tqdm import tqdm
from glob import glob
import os

def img_resize(W, D):
    file_list = glob('./img/raw/*.jpg')
    W, D = int(W), int(D)
    for source_image in tqdm(file_list):
        image = Image.open(source_image)
        filename = os.path.basename(source_image).split('.')[0]
        resize_image = image.resize((W, D))
        resize_image.save('./img/resize/{}_l.jpg'.format(filename), "JPEG", quality=95 )


if __name__ == '__main__':
    W, D = input("차원을 입력해주세요 W H : ").split()
    img_resize(W, D)
