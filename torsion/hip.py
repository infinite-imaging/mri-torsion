import numpy as np
import math
from scipy.optimize import curve_fit
from skimage import measure

from torsion.utils import (write_image, get_contour_points, find_notch,
                            rotate_mask_dorsal_pts, transform_pt, get_centroid,
                            get_vector, length, round_to_int, get_contour)
from torsion.bresenham import bresenhamline


def sphere_fit(sp_x, sp_y, sp_z):
  
    sp_x = np.array(sp_x)
    sp_y = np.array(sp_y)
    sp_z = np.array(sp_z)
    A = np.zeros((len(sp_x), 4))
    A[:, 0] = sp_x * 2
    A[:, 1] = sp_y * 2
    A[:, 2] = sp_z * 2
    A[:, 3] = 1

    f = np.zeros((len(sp_x), 1))
    f[:, 0] = (sp_x * sp_x) + (sp_y * sp_y) + (sp_z * sp_z)
    C, _, _, _ = np.linalg.lstsq(A, f)

    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = math.sqrt(t)

    return radius, C[0], C[1], C[2]


def draw_sphere(mask, r, center, z_ratio):
    for z in range(mask.shape[0]): 
        for x in range(center[2] - int(r) - 2,
                       center[2] + int(r) + 2):
            if r**2 - (x - center[2])**2 - (z_ratio * (z - center[0]))**2 >= 0:
                y = int(
                    round(
                        math.sqrt(r**2 - (x - center[2])**2 -
                                  (z_ratio * (z - center[0]))**2)))
                if y < mask.shape[1]:
                    mask[z, center[1] + y, x] = 5
                    mask[z, center[1] - y, x] = 5

        for y in range(center[1] - int(r) - 2,
                       center[1] + int(r) + 2):
            if r**2 - (y - center[1])**2 - (z_ratio * (z - center[0]))**2 >= 0:
                x = int(
                    round(
                        math.sqrt(r**2 - (y - center[1])**2 -
                                  (z_ratio * (z - center[0]))**2)))
                if x < mask.shape[2]:
                    mask[z, y, center[2] + x] = 5
                    mask[z, y, center[2] - x] = 5


def draw_circle(mask, layer, r, center):
    for x in range(center[1] - int(r) - 2,
                   center[1] + int(r) + 2):
        if r**2 - (x - center[1])**2 >= 0:
            y = int(round(math.sqrt(r**2 - (x - center[1])**2)))
            if y < mask.shape[1]:
                mask[layer, center[0] + y, x] = 5
                mask[layer, center[0] - y, x] = 5

    for y in range(center[0] - int(r) - 2,
                   center[0] + int(r) + 2):
        if r**2 - (y - center[0])**2 >= 0:
            x = int(round(math.sqrt(r**2 - (y - center[0])**2)))
            if x < mask.shape[2]:
                mask[layer, y, center[1] + x] = 5
                mask[layer, y, center[1] - x] = 5


def pts_on_circle(mask, r, center):
    rt = False
    for x in range(max(0, center[1] - int(r) - 2),
                   min(mask.shape[1], center[1] + int(r) + 2)):
        temp = r**2 - (x - center[1])**2
        if temp > 0 and (
                mask[max(0, int(round(center[0] - math.sqrt(temp)))), x] != 0
                or mask[min(int(round(center[0] + math.sqrt(temp))), mask.shape[0]-1), x] != 0):
            rt = True
            break
    if rt:
        return rt
    else:
        for y in range(max(0, center[0] - int(r) - 2),
                       min(mask.shape[0], center[0] + int(r) + 2)):
            temp = r**2 - (y - center[0])**2
            if temp > 0 and (
                    mask[y, min(0, int(round(center[1] - math.sqrt(temp))))] != 0
                    or mask[y, max(mask.shape[1]-1, int(round(center[1] + math.sqrt(temp))))] != 0):
                rt = True
                break
    return rt


def contour_femoral_neck(mask, contour, layer_selected, center, r):
    mask_new = mask[layer_selected].copy()
    rotated_mask, angle1 = rotate_mask_dorsal_pts(mask_new,
                                                  get_centroid(mask_new))
    rotated_mask, angle2 = rotate_mask_dorsal_pts(rotated_mask,
                                                  get_centroid(rotated_mask))
    angle = angle1 + angle2
    notch_rot = find_notch(rotated_mask,
                           percentage=0.8,
                           thresh=5,
                           break_after_first=True)
    rot_offset = np.array([
        _rot_dim - _orig_dim
        for _rot_dim, _orig_dim in zip(rotated_mask.shape, mask_new.shape)
    ])
    rot_center = (int(
        (rotated_mask.shape[0] - 1) / 2), int((rotated_mask.shape[1] - 1) / 2))
    if angle == 0:
        notch = notch_rot
    else:
        notch = transform_pt(notch_rot,
                             rot_center,
                             -angle,
                             offset=-rot_offset / 2)

    a = length(get_vector(notch, [center[1], center[2]])) / r
    r_1 = (-0.1 + a) * r
    r_2 = (0.1 + a) * r

    mask_new = mask[layer_selected].copy()

    for w in range(mask.shape[2] - 1):
        if r_2**2 - (w - center[2])**2 >= 0:
            y_b = int(round(math.sqrt(r_2**2 - (w - center[2])**2)))
            if center[1] + y_b < mask.shape[1]:
                contour[layer_selected, center[1] + y_b:, w] = 0
                mask_new[center[1] + y_b:, w] = 0
            if center[1] - y_b > 0:
                contour[layer_selected, :center[1] - y_b, w] = 0
                mask_new[:center[1] - y_b, w] = 0
        else:
            contour[layer_selected, :, w] = 0
            mask_new[:, w] = 0

        if r_1**2 - (w - center[2])**2 >= 0:
            y_c = int(round(math.sqrt(r_1**2 - (w - center[2])**2)))
            if center[1] - y_c > 0 and center[1] + y_c < mask.shape[1]:
                contour[layer_selected, center[1] - y_c:center[1] + y_c, w] = 0
                mask_new[center[1] - y_c:center[1] + y_c, w] = 0

    return mask_new, contour, r_1, r_2


def calc_hip(mask, z_ratio, path_out=None, path_out_ball=None, mark_points: bool = False):
    contour_pts = get_contour_points(mask)

    layer_high = np.amax(contour_pts[0])
    com_high = get_centroid(mask[layer_high - 1])
    layer_low = layer_high
    while mask[layer_low - 1, com_high[0], com_high[1]] != 0:
        layer_low -= 1

    if com_high[1] > int((mask.shape[2] - 1) / 2):
        femur_left = True
    else:
        femur_left = False

    contour = np.zeros(mask.shape).astype(np.uint8)
    for k in range(len(mask)):
        contour[k] = get_contour(mask[k])

    correct_x = np.ndarray(0)
    correct_y = np.ndarray(0)
    correct_z = np.ndarray(0)
    correct_z_new = np.ndarray(0)
    if not femur_left:
        for x in range(np.amin(contour_pts[2]), np.amax(contour_pts[2]) + 1):
            correct = np.nonzero(contour[layer_low:, :com_high[0],
                                         com_high[1]:])
            correct = np.nonzero(contour[layer_low:, :(x + com_high[0] -
                                                       com_high[1]), x])
            correct_x = np.append(correct_x, np.full(correct[0].size, x))
            correct_y = np.append(correct_y, correct[1])
            correct_z = np.append(correct_z,
                                  (correct[0] + layer_low) * z_ratio)
    else:
        for x in range(np.amin(contour_pts[2]), np.amax(contour_pts[2]) + 1):
            correct = np.nonzero(contour[layer_low:, :(-x + com_high[0] +
                                                       com_high[1]), x])
            correct_x = np.append(correct_x, np.full(correct[0].size, x))
            correct_y = np.append(correct_y, correct[1])
            correct_z = np.append(correct_z,
                                  (correct[0] + layer_low) * z_ratio)

    r, x0, y0, z0 = sphere_fit(correct_x, correct_y, correct_z)
    z0 = z0 / z_ratio
    center = round_to_int((z0[0], y0[0], x0[0]))

    if mark_points:
        mask[center[0], center[1], center[2]] = 5
    if path_out is not None:
        write_image(mask, path_out)

    layer_selected = None

    for n in range(mask.shape[0] - 1, 0, -1):
        if len(measure.find_contours(mask[n], 0.8)) == 1 and \
                pts_on_circle(mask[n], r*2, [center[1], center[2]]):
            layer_selected = n
            break

    mask_new, contour, r_1, r_2 = contour_femoral_neck(mask, contour,
                                                       layer_selected, center,
                                                       r)
    while np.count_nonzero(mask_new == 1) < 65:
        layer_selected = layer_selected - 1
        mask_new, contour, r_1, r_2 = contour_femoral_neck(
            mask, contour, layer_selected, center, r)

    center_fn = get_centroid(mask_new)

    contour_pts_l = np.nonzero(contour[layer_selected])

    distance_center = (contour_pts_l[0] -
                       center_fn[0])**2 + (contour_pts_l[1] - center_fn[1])**2
    r_3 = math.sqrt(np.median(distance_center)) * 1.5

    for w in range(mask.shape[2] - 1):
        if r_3**2 - (w - center_fn[1])**2 >= 0:
            y_b = int(round(math.sqrt(r_3**2 - (w - center_fn[1])**2)))
            if center_fn[0] + y_b < mask.shape[1]:
                contour[layer_selected, center_fn[0] + y_b:, w] = 0
            if center_fn[0] - y_b > 0:
                contour[layer_selected, :center_fn[0] - y_b, w] = 0
        else:
            contour[layer_selected, :, w] = 0

    contour_pts_l = np.nonzero(contour[layer_selected])

    diff = np.ediff1d(contour_pts_l[0])
    ind_gap = np.argsort(diff)[-1] + 1

    def g(x, m):
        return m * x

    if diff.max() == 1:
        popt, _ = curve_fit(g, contour_pts_l[1] - center[2],
                            contour_pts_l[0] - center[1])
        m_new = popt[0]
    else:
        popt1, _ = curve_fit(g, contour_pts_l[1][:ind_gap] - center[2],
                             contour_pts_l[0][:ind_gap] - center[1])
        popt2, _ = curve_fit(g, contour_pts_l[1][ind_gap:] - center[2],
                             contour_pts_l[0][ind_gap:] - center[1])
        m_new = np.mean([popt1[0], popt2[0]])

    if not femur_left:
        end = round_to_int(((-80) * m_new + center[1], -80 + center[2]))
    else:
        end = round_to_int((80 * m_new + center[1], 80 + center[2]))

    if mark_points:
        draw_sphere(mask, r, center, z_ratio=z_ratio)
        draw_circle(mask, layer_selected, r_1, center[1:])
        draw_circle(mask, layer_selected, r_2, center[1:])
        draw_circle(mask, layer_selected, r_3, center_fn)

        draw_circle(mask, layer_selected - 1, r_1, center[1:])
        draw_circle(mask, layer_selected - 1, r_2, center[1:])
        draw_circle(mask, layer_selected - 1, r_3, center_fn)

        line = bresenhamline([(center[1], center[2])], end, max_iter=-1)
        for u in range(len(line)):
            mask[:, int(line[u, 0]), int(line[u, 1])] = 5

    if mark_points:
        mask = mask + contour
    
    if path_out is not None:
        write_image(mask, path_out)
    return mask, center, np.append(layer_selected, end)
