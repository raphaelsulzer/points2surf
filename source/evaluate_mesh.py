import numpy as np
from source.libmesh import check_mesh_contains

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def eval_mesh(gt_file, recon_mesh, rotate=0):

    # print("gt_file: ",gt_file)

    occ = np.load(gt_file)
    occ_points = occ["points"]
    if(rotate):
        R=np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]],dtype=np.float32)
        occ_points=np.matmul(occ_points,R)
    else:
        a=5

    gt_occ = occ["occupancies"]
    gt_occ = np.unpackbits(gt_occ)[:occ_points.shape[0]]
    gt_occ = gt_occ.astype(np.bool)

    try:
        recon_occ = check_mesh_contains(recon_mesh, occ_points)
        # print("recon: ", recon_occ.shape)
        # print("recon: ", recon_occ.dtype)
        # print("gt: ", gt_occ.shape)
        # print("gt: ", gt_occ.dtype)
        return compute_iou(gt_occ, recon_occ)
    except:
        print("WARNING: Could not calculate IoU for mesh ", gt_file)
        return 0.0






