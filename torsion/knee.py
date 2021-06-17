import numpy as np
from torsion.utils import (write_image, get_contour_points,
                            get_dorsal_mask_pt, find_notch,
                            get_layer_with_biggest_convex_area, rotate_pt,
                            rotate_mask_vec_parallel, rotate_mask_dorsal_pts,
                            transform_pt, get_centroid, get_vector, length,
                            round_to_int)
from torsion.bresenham import bresenhamline


def rotate_tibia(mask):
    contour = get_contour_points(mask)
    d = 0
    for k in range(len(contour[0])):
        for l in range(len(contour[0])):
            y1 = contour[0][k]
            x1 = contour[1][k]
            y2 = contour[0][l]
            x2 = contour[1][l]

            if x1 < x2:
                vec = get_vector((y1, x1), (y2, x2))
            else:
                vec = get_vector((y2, x2), (y1, x1))
            if length(vec) > d:
                d = length(vec)
                line = vec
    return rotate_mask_vec_parallel(mask, line, np.array([0, 1]))


def num_mask_points_on_line(mask, start, end, thresh_pt=None):
    pts_on_line = bresenhamline([start], end, -1).astype(np.uint16)

    if thresh_pt is not None:
        if start[1] < thresh_pt[1]:
            pts_on_line = np.array(
                [pt for pt in pts_on_line if pt[1] >= thresh_pt[1]])
        else:
            pts_on_line = np.array(
                [pt for pt in pts_on_line if pt[1] <= thresh_pt[1]])

    sum_val = 0
    for pt in pts_on_line:
        sum_val += mask[pt[0], pt[1]]

    return sum_val


def shrink_points_to_mask(mask, start_pt, end_pt):

    points_on_line = bresenhamline([start_pt], end_pt, -1)

    shrinked_start, shrinked_end = None, None

    for pt in points_on_line:
        if mask[int(pt[0]), int(pt[1])]:
            shrinked_start = pt
            break

    for pt in points_on_line[::-1]:
        if mask[int(pt[0]), int(pt[1])]:
            shrinked_end = pt
            break

    return shrinked_start, shrinked_end


def calc_knee(bone,
              mask,
              path_out=None,
              path_out_ro=None,
              start_pt=None,
              thresh=2,
              step_size=1,
              return_notch=True,
              mark_points: bool =False):

    layer = get_layer_with_biggest_convex_area(mask)
    mask_l = mask[layer]

    if bone == "tibia":
        notch = find_notch(mask_l, percentage=0.5, thresh=2)
        if notch[0] is None:
            rotated_mask, ang1 = rotate_tibia(mask_l)
            rot_thresh = find_notch(rotated_mask, percentage=0.5, thresh=2)
            if rot_thresh[0] is None:
                rot_thresh = get_centroid(rotated_mask)
            rotated_mask, ang2 = rotate_mask_dorsal_pts(
                rotated_mask, rot_thresh)
            angle = ang1 + ang2
            notch_rot = find_notch(rotated_mask, percentage=0.5, thresh=2)
        else:
            rotated_mask, angle = rotate_mask_dorsal_pts(mask_l, notch)
    elif bone == "femur":
        notch = find_notch(mask_l, percentage=0.7, thresh=1)
        if notch[0] is None:
            rotated_mask, ang1 = rotate_tibia(mask_l)
            rot_thresh = find_notch(rotated_mask, percentage=0.7, thresh=2)
            if rot_thresh[0] is None:
                rot_thresh = get_centroid(rotated_mask)
            rotated_mask, ang2 = rotate_mask_dorsal_pts(
                rotated_mask, rot_thresh)
            angle = ang1 + ang2
            notch_rot = find_notch(rotated_mask, percentage=0.7, thresh=2)
        else:
            rotated_mask, angle = rotate_mask_dorsal_pts(mask_l, notch)
    else:
        raise ValueError

    if path_out_ro is not None:
        write_image(rotated_mask, path_out_ro)

    rot_offset = np.array([
        _rot_dim - _orig_dim
        for _rot_dim, _orig_dim in zip(rotated_mask.shape, mask[layer].shape)
    ])
    rot_center = (int(
        (rotated_mask.shape[0] - 1) / 2), int((rotated_mask.shape[1] - 1) / 2))

    if notch[0] is not None:
        notch_rot = transform_pt(notch, (int(
            (mask_l.shape[0] - 1) / 2), int((mask_l.shape[1] - 1) / 2)),
                                 angle,
                                 offset=rot_offset / 2)
    else:
        notch = transform_pt(notch_rot,
                             rot_center,
                             -angle,
                             offset=-rot_offset / 2)

    if start_pt is None:
        start_pt = get_dorsal_mask_pt(rotated_mask)
    else:
        start_pt = transform_pt(start_pt, (int(
            (mask_l.shape[0] - 1) / 2), int((mask_l.shape[1] - 1) / 2)),
                                angle,
                                offset=rot_offset / 2)

    if start_pt[1] < notch_rot[1]:
        x_end = rotated_mask.shape[1] - 1
        rot_dir = 1
    else:
        x_end = 0
        rot_dir = -1
    end_pt = (start_pt[0], x_end)

    while num_mask_points_on_line(rotated_mask, start_pt, end_pt,
                                  notch_rot) < thresh:
        end_pt = rotate_pt(start_pt, end_pt, step_size * rot_dir)

    _, final_end_pt = shrink_points_to_mask(rotated_mask, start_pt, end_pt)

    end_pt_orig = transform_pt(final_end_pt,
                               rot_center,
                               -angle,
                               offset=-rot_offset / 2)
    start_pt_orig = transform_pt(start_pt,
                                 rot_center,
                                 -angle,
                                 offset=-rot_offset / 2)

    start_pt_orig = round_to_int(start_pt_orig)
    end_pt_orig = round_to_int(end_pt_orig)
    notch = round_to_int(notch)

    if mark_points:
        line = bresenhamline([start_pt_orig], end_pt_orig, max_iter=-1)
        for m in range(len(line)):
            mask_l[int(line[m, 0]), int(line[m, 1])] = 3

    start_pt_orig = (layer, start_pt_orig[0], start_pt_orig[1])
    end_pt_orig = (layer, end_pt_orig[0], end_pt_orig[1])
    notch = (layer, notch[0], notch[1])

    if mark_points:
        mask[start_pt_orig] = 5
        mask[end_pt_orig] = 5
        mask[notch] = 5

    if path_out is not None:
        write_image(mask, path_out)

    if return_notch:
        return mask, start_pt_orig, end_pt_orig, notch
    else:
        return mask, start_pt_orig, end_pt_orig