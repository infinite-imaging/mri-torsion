from skimage.measure import regionprops, label
import numpy as np
from torsion.utils import get_centroid, write_image
from torsion.bresenham import bresenhamline


def get_layer_with_largest_diameter(mask):


    diameter = np.zeros(mask.shape[0])
    for k in range(len(mask)):
        if len(np.nonzero(mask[k])[0]) != 0:
            props = regionprops(label(mask[k]))
            if props.__len__() > 1:
                i_biggest = 0
                for i in range(props.__len__()):
                    if props[i].equivalent_diameter > props[
                            i_biggest].equivalent_diameter:
                        i_biggest = i
                diameter[k] = props[i_biggest].equivalent_diameter
            else:
                diameter[k] = props[0].equivalent_diameter
    indices = np.argsort(diameter)
    return indices[-1]


def calc_ankle(mask_t, mask_f, out_t=None, mark_points: bool = False):
    layer = get_layer_with_largest_diameter(mask_t)

    com_tibia = get_centroid(mask_t[layer])
    com_fibula = get_centroid(mask_f[layer])

    mask = mask_t + mask_f

    if mark_points:
        line = bresenhamline([com_tibia], com_fibula, max_iter=-1)
        for k in range(len(line)):
            mask[layer, int(line[k, 0]), int(line[k, 1])] = 3

    com_tibia = (layer, com_tibia[0], com_tibia[1])
    com_fibula = (layer, com_fibula[0], com_fibula[1])

    if mark_points:
        mask[com_tibia] = 5
        mask[com_fibula] = 5

    if out_t is not None:
        write_image(mask, out_t)

    return mask, com_tibia, com_fibula