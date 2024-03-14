# https://github.com/melqkiades/deep-wetlands/tree/master - откуда воровать код

# TODO: подготовить набор данных
#       скачать pytorch
import rasterio
import matplotlib.pyplot as plt
from scripts.bands_scripts import *

if __name__ == "__main__":
    data_rgb = get_image_patches(r'D:\Users\david.trufanov\PycharmProjects\pythonProject\train_data\0_rgb.tiff')
    data_ndwi = get_image_patches(r'D:\Users\david.trufanov\PycharmProjects\pythonProject\train_data\0_ndwi.tiff')
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].imshow(prepare_rgb(data_rgb[0, 0]))
    axs[1].imshow(prepare_ndwi(data_ndwi[0, 0][0]))
    plt.show()
