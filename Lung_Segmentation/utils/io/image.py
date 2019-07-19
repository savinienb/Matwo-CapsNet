
from utils.io.common import create_directories_for_file_name
import SimpleITK as sitk
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def write_nd_np(img, path):
    write(sitk.GetImageFromArray(np.transpose(img, (1, 2, 3, 0))), path)

def write_np(img, path):
    if len(img.shape) == 4:
        write(sitk.GetImageFromArray(np.transpose(img, (1, 2, 3, 0))), path)
    else:
        write(sitk.GetImageFromArray(img), path)


def write_np_colormask(img, path, num_labels=8):
    create_directories_for_file_name(path)
    im_out=sitk.GetArrayFromImage(img)
    im_out=np.asarray(im_out,dtype=np.float32)
    im_out -= np.amin(im_out)
    im_out /= (num_labels-1)
    #im_out /= np.amax(im_out)
    plt.imsave(path, im_out, cmap='tab20')

def write(img, path):
    """
    Write a volume to a file path.

    :param img: the volume
    :param path: the target path
    :return:
    """
    create_directories_for_file_name(path)
    writer = sitk.ImageFileWriter()
    writer.Execute(img, path, True)


def write_np_rgb(img, path):
    assert(img.shape[0] == 3)
    rgb_components = [sitk.GetImageFromArray(img[i, :, :]) for i in range(img.shape[0])]
    filter = sitk.ComposeImageFilter()
    rgb = filter.Execute(rgb_components[0], rgb_components[1], rgb_components[2])
    write(rgb, path)


def read(path, sitk_pixel_type=sitk.sitkInt16):
    image = sitk.ReadImage(path, sitk_pixel_type)
    x = image.GetNumberOfComponentsPerPixel()
    # TODO every sitkVectorUInt8 image is converted to have 3 channels (RGB) -> we may not want that
    if sitk_pixel_type == sitk.sitkVectorUInt8 and x == 1:
        image_single = sitk.VectorIndexSelectionCast(image)
        image = sitk.Compose(image_single, image_single, image_single)
    return image


def read_meta_data(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    return reader
