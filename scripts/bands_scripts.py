import rasterio
import numpy as np
import os
from skimage.util import view_as_blocks

ALPHA = 0.06  # Contrast control
BETA = 0  # Brightness control


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band - band_min) / (band_max - band_min))


def brighten(band):
    return np.clip(ALPHA * band + BETA, 0, 255)


def prepare_rgb(img: np.array) -> np.array:
    ''' RGB: from (3,x,y) to (x,y,3)'''
    img = np.moveaxis(img, 0, -1)
    x_std = (img - img.min()) / (img.max() - img.min())
    img = x_std * (255 - 0) + 0
    return np.uint8(img)


def prepare_ndwi(img: np.array) -> np.array:
    '''from [-1,1] to [0,255]'''
    x_std = (img - img.min()) / (img.max() - img.min())
    x_unscaled = x_std * (255 - 0) + 0
    return np.uint8(x_unscaled)


def get_path_to_image(in_dir):
    image_dir = os.path.join(in_dir, 'GRANULE')
    tree = os.walk(image_dir)
    image_dir = list(filter(lambda x: 'IMG_DATA' in x[0], list(tree)))
    match len(image_dir):
        case 1:
            bands_dir = [os.path.join(image_dir[0][0], i) for i in image_dir[0][2]]
        case 4:
            bands_dir = [os.path.join(image_dir[1][0], i) for i in image_dir[1][2]]
    return bands_dir


def get_image_patches(image_path: str, patch_size=1000):
    full_image = rasterio.open(image_path).read()[:, :10000, :10000]
    match full_image.shape[0]:
        case 1:
            #full_image = prepare_ndwi(full_image)
            return view_as_blocks(full_image, block_shape=(1, patch_size, patch_size))[0]
        case 3:
            #full_image = prepare_rgb(full_image)
            return view_as_blocks(full_image, block_shape=(3, patch_size, patch_size))[0]


def get_train_data(raw_data_dir, train_data_out):
    for index, image in enumerate(os.listdir(raw_data_dir)):
        get_rgb(os.path.join(raw_data_dir, image), train_data_out, index)
        get_ndwi(os.path.join(raw_data_dir, image), train_data_out, index)


def get_rgb(in_dir, out_dir, name):
    bands_dir = get_path_to_image(in_dir)
    red = rasterio.open([i for i in bands_dir if 'B04' in i][0])
    green = rasterio.open([i for i in bands_dir if 'B03' in i][0])
    blue = rasterio.open([i for i in bands_dir if 'B02' in i][0])
    rgb_tif = rasterio.open(os.path.join(out_dir, f'{name}_rgb.tiff'), 'w', driver='Gtiff',
                            width=green.width, height=green.height,
                            count=3,
                            crs=green.crs,
                            transform=green.transform,
                            dtype='float32')

    red = normalize(brighten(red.read(1).astype('float32')))
    green = normalize(brighten(green.read(1).astype('float32')))
    blue = normalize(brighten(blue.read(1).astype('float32')))
    rgb_tif.write(red, 1)
    rgb_tif.write(green, 2)
    rgb_tif.write(blue, 3)
    rgb_tif.close()


def get_ndwi(in_dir, out_dir, name):
    bands_dir = get_path_to_image(in_dir)
    band_green = rasterio.open([i for i in bands_dir if 'B03' in i][0])
    band_nir = rasterio.open([i for i in bands_dir if 'B08' in i][0])
    ndwi_tif = rasterio.open(os.path.join(out_dir, f'{name}_ndwi.tiff'), 'w', driver='Gtiff',
                             width=band_green.width, height=band_green.height,
                             count=1,
                             crs=band_green.crs,
                             transform=band_green.transform,
                             dtype='float32')
    band_green = band_green.read(1).astype('float64')
    band_nir = band_nir.read(1).astype('float64')
    ndwi = (band_green - band_nir) / (band_green + band_nir)
    ndwi_tif.write(ndwi, 1)
    ndwi_tif.close()


if __name__ == "__main__":
    raw_dir = r'D:\Users\david.trufanov\PycharmProjects\pythonProject\raw_data'
    train_dir = r'D:\Users\david.trufanov\PycharmProjects\pythonProject\train_data'
    get_train_data(raw_dir, train_dir)
