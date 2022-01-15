import numpy as np
from scipy.spatial import cKDTree


def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor

def add_non_uniform(pointcloud_dict):
    # t0 = time.time()
    res = 100

    points = pointcloud_dict['points']
    normals = pointcloud_dict['normals']
    gt_normals = pointcloud_dict['gt_normals']
    sensors = pointcloud_dict['sensor_pos']

    npoints = normalize_3d_coordinate(points, padding=0.1)
    pindex = (npoints * res).astype(int)

    pgrid = np.zeros(shape=(res, res, res), dtype=bool)

    # apply buffering / dilation, with 5x5x5 kernel, and active the pgrid voxels
    # this could maybe be sped up by using openCV: dilation = cv2.dilate(img,kernel,iterations = 1)

    temp = np.arange(-2, 3)
    kernel = np.array(np.meshgrid(temp, temp, temp)).T.reshape(-1, 3)  # 5x5x5 Kernel
    for k in kernel:
        pgrid[pindex[:, 0] + k[0], pindex[:, 1] + k[1], pindex[:, 2] + k[2]] = True

    sensor_vecs = sensors - points
    sensor_vecs = sensor_vecs / np.linalg.norm(sensor_vecs, axis=1)[:, np.newaxis]

    n = 50
    steps = np.expand_dims(np.linspace(0.01, 0.5, n), axis=1)

    ## inside:
    m = 2
    npoints = np.repeat(points, m, axis=0)
    ident = np.arange(points.shape[0])
    ident = np.repeat(ident, m, axis=0)
    ident = np.expand_dims(ident, axis=1)
    nsensors = np.repeat(sensor_vecs, m, axis=0)
    nsteps = np.tile(steps[:m], [points.shape[0], 3])
    in_points = npoints - nsteps * nsensors
    in_points = np.concatenate((in_points, ident), axis=1)

    nin_points = normalize_3d_coordinate(in_points[:, :3], padding=0.1)
    iindex = (nin_points * res).astype(int)
    igrid = np.zeros(shape=(res, res, res), dtype=int)
    # if a voxel includes more than one los_points, this will simply choose the first los_point in the list!
    igrid[iindex[:, 0], iindex[:, 1], iindex[:, 2]] = np.arange(iindex.shape[0])
    selected_iindex = igrid[igrid > 0]
    in_points = in_points[selected_iindex]

    ## outside:
    npoints = np.repeat(points, n, axis=0)
    ident = np.arange(points.shape[0])
    ident = np.repeat(ident, n, axis=0)
    ident = np.expand_dims(ident, axis=1)
    nsensors = np.repeat(sensor_vecs, n, axis=0)
    nsteps = np.tile(steps, [points.shape[0], 3])
    los_points = npoints + nsteps * nsensors
    los_points = np.concatenate((los_points, ident), axis=1)

    nlos_points = normalize_3d_coordinate(los_points[:, :3], padding=0.1)
    lindex = (nlos_points * res).astype(int)

    lgrid = np.zeros(shape=(res, res, res), dtype=int)
    # if a voxel includes more than one los_points, this will simply choose the first los_point in the list!
    lgrid[lindex[:, 0], lindex[:, 1], lindex[:, 2]] = np.arange(lindex.shape[0])

    # if there is a (buffered) point, keep the los_point
    active = lgrid * pgrid
    selected_lindex = active[active > 0]
    los_points = los_points[selected_lindex]

    ### put everything together
    cident = np.zeros(shape=(points.shape[0], 2))
    ins = np.concatenate((np.ones(shape=(in_points.shape[0], 1)), np.zeros(shape=(in_points.shape[0], 1))), axis=1)
    out = np.concatenate((np.zeros(shape=(los_points.shape[0], 1)), np.ones(shape=(los_points.shape[0], 1))), axis=1)
    cident = np.concatenate((cident,
                             ins,
                             out))

    sensor_vecs = np.concatenate((sensor_vecs,
                                  sensor_vecs[in_points[:, 3].astype(int)],
                                  sensor_vecs[los_points[:, 3].astype(int)]))
    normals = np.concatenate((normals,
                              normals[in_points[:, 3].astype(int)],
                              normals[los_points[:, 3].astype(int)]))
    gt_normals = np.concatenate((gt_normals,
                                 gt_normals[in_points[:, 3].astype(int)],
                                 gt_normals[los_points[:, 3].astype(int)]))

    points = np.concatenate((points,
                             in_points[:, :3],
                             los_points[:, :3]))
    # points = np.concatenate((points, cident), axis=1)

    # print("time: ", time.time() - t0)

    data = {
        'points': points.astype(np.float32),
        'ident': cident.astype(np.float32),
        'normals': normals.astype(np.float32),
        'gt_normals': gt_normals.astype(np.float32),
        'sensor_pos': sensor_vecs.astype(np.float32),
    }

    return data



def add_uniform_neighborhood(pointcloud_dict,workers):
    # make los-points that are close ( <= average neighborhoodsize) to end point of los

    # take mean of this vector: factor = np.array(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pc)).mean()
    # make sensor vector a unit vector and then do:
    # sampled_point = points + norm_sensor_vec * factor

    points = pointcloud_dict['points']
    normals = pointcloud_dict['normals']
    gt_normals = pointcloud_dict['gt_normals']
    sensors = pointcloud_dict['sensor_pos']

    # get the factor for where to put the point
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(points)
    # mean_dist = np.array(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pc)).mean()
    tree = cKDTree(points)
    mean_dist = tree.query(points, k=2, n_jobs=workers)[0][:, 1].mean()

    # add the point identifier
    ident = np.concatenate((np.zeros(shape=(points.shape[0], 2), dtype=np.float32),
                           np.concatenate((np.zeros(shape=(points.shape[0], 1), dtype=np.float32),
                                           np.ones(shape=(points.shape[0], 1), dtype=np.float32)), axis=1),
                           np.concatenate((np.ones(shape=(points.shape[0], 1), dtype=np.float32),
                                           np.zeros(shape=(points.shape[0], 1), dtype=np.float32)), axis=1)))

    # make the sensor vec
    sensor_vec = sensors - points
    # normalize the sensors
    sensor_vec_norm = sensor_vec / np.linalg.norm(sensor_vec, axis=1)[:, np.newaxis]

    opoints = points[:, :3] + mean_dist * sensor_vec_norm
    ipoints = points[:, :3] - mean_dist * sensor_vec_norm

    points = np.concatenate((points, opoints, ipoints))
    normals = np.concatenate((normals, normals, normals))
    gt_normals = np.concatenate((gt_normals, gt_normals, gt_normals))
    sensors = np.concatenate((sensors, sensors, sensors))

    data = {
        'points': points,
        'ident': ident,
        'normals': normals,
        'gt_normals': gt_normals,
        'sensor_pos': sensors,
    }

    return data