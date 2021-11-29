import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import trimesh

from source.base import utils
from source import sdf

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

def load_points_and_sensor_information(point_filename):

    data_dict = {}

    # get the points
    data = np.load(point_filename)
    data_dict["points"] = data["points"].astype(np.float32)
    data_dict["normals"] = data["normals"].astype(np.float32)
    data_dict["gt_normals"] = data["gt_normals"].astype(np.float32)
    data_dict["sensor_pos"] = data["sensors"].astype(np.float32)

    ### shouldn't do that here, because there are transformations applied to the points, so I should create the vectors after the transformation?!
    # data_dict["sensor_vec"] = (data_dict["sensor_pos"] - data_dict["points"]).astype(np.float32)
    # data_dict["sensor_vec_norm"] = data_dict["sensor_vec"] / np.linalg.norm(data_dict["sensor_vec"], axis=1)[:, np.newaxis]


    # make the auxiliary points


    return data_dict




def load_shape(point_filename, imp_surf_query_filename, imp_surf_dist_filename,
               query_grid_resolution=None, epsilon=None, sensor=None):
    """
    do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
    so modifying the points would make the kdtree give incorrect results
    :param point_filename:
    :param imp_surf_query_filename:
    :param imp_surf_dist_filename:
    :param query_grid_resolution: if not None: create query points at grid of this resolution
    :param epsilon:
    :return:
    """

    # mmap_mode = 'r'

    data_dict = load_points_and_sensor_information(point_filename)
    ori_shape = data_dict["points"].shape[0]
    if(sensor=="grid"):
        data_dict = add_non_uniform(data_dict)

    pts_np = data_dict['points']

    # otherwise KDTree construction may run out of recursions
    leaf_size = 1000
    sys.setrecursionlimit(int(max(1000, round(pts_np.shape[0]/leaf_size))))
    kdtree = spatial.cKDTree(pts_np, leaf_size)

    if imp_surf_dist_filename is not None:
        imp_surf_dist_ms = np.load(imp_surf_dist_filename)
        if imp_surf_dist_ms.dtype != np.float32:
            print('Warning: imp_surf_dist_ms must be converted to float32')
            imp_surf_dist_ms = imp_surf_dist_ms.astype(np.float32)
    else:
        imp_surf_dist_ms = None

    if imp_surf_query_filename is not None:
        imp_surf_query_point_ms = np.load(imp_surf_query_filename)
        if imp_surf_query_point_ms.dtype != np.float32:
            print('Warning: imp_surf_query_point_ms must be converted to float32')
            imp_surf_query_point_ms = imp_surf_query_point_ms.astype(np.float32)
    elif query_grid_resolution is not None:
        # get query points near at grid nodes near the point cloud
        # get patches for those query points
        ### MINE: this is where I need to restrict the query point search to the original pointcloud without auxiliary points
        imp_surf_query_point_ms = sdf.get_voxel_centers_grid_smaller_pc(
            pts=pts_np[:ori_shape], grid_resolution=query_grid_resolution,
            distance_threshold_vs=epsilon)
    else:
        imp_surf_query_point_ms = None

    return Shape(data=data_dict,
        pts=pts_np, kdtree=kdtree,
        imp_surf_query_point_ms=imp_surf_query_point_ms, imp_surf_dist_ms=imp_surf_dist_ms)


class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=False, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + \
                                     min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset
        # where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end),
                                                size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count


class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + \
                                     min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(
            sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape:
    def __init__(self, data, pts, kdtree,
                 imp_surf_query_point_ms, imp_surf_dist_ms):
        self.data = data
        self.pts = pts
        self.kdtree = kdtree
        self.imp_surf_query_point_ms = imp_surf_query_point_ms
        self.imp_surf_dist_ms = imp_surf_dist_ms


class Cache:
    def __init__(self, capacity, loader, loadfunc):
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):

    def __init__(self, opt, root, scan, shape_list_filename, points_per_patch, patch_radius, patch_features, epsilon,
                 seed=None, identical_epochs=False, center='point',
                 cache_capacity=1, point_count_std=0.0,
                 pre_processed_patches=False, query_grid_resolution=None,
                 sub_sample_size=500, reconstruction=False, uniform_subsample=False,
                 num_workers=1, n_classes=1, shapes_per_class=2):

        # initialize parameters
        self.opt = opt
        self.root = root
        self.scan = scan
        self.shape_list_filename = shape_list_filename
        self.patch_features = patch_features
        self.points_per_patch = points_per_patch
        self.patch_radius = patch_radius
        self.identical_epochs = identical_epochs
        self.pre_processed_patches = pre_processed_patches
        self.center = center
        self.point_count_std = point_count_std
        self.seed = seed
        self.query_grid_resolution = query_grid_resolution
        self.sub_sample_size = sub_sample_size
        self.reconstruction = reconstruction
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.uniform_subsample = uniform_subsample

        self.include_connectivity = False
        self.include_imp_surf = False
        self.include_p_index = False
        self.include_patch_pts_ids = False
        for pfeat in self.patch_features:
            if pfeat == 'imp_surf':
                self.include_imp_surf = True
            elif pfeat == 'imp_surf_magnitude':
                self.include_imp_surf = True
            elif pfeat == 'imp_surf_sign':
                self.include_imp_surf = True
            elif pfeat == 'p_index':
                self.include_p_index = True
            elif pfeat == 'patch_pts_ids':
                self.include_patch_pts_ids = True
            else:
                raise ValueError('Unknown patch feature: %s' % pfeat)

        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        temp = []
        self.shape_names = []
        classes = os.listdir(self.root)
        classes = classes[:n_classes]
        # remove class x
        if 'x' in classes: classes.remove('x')
        for c in classes:
            # with open(os.path.join(self.root, c, "p2s", self.shape_list_filename)) as f:
            with open(os.path.join(self.root, c, "convonet", scan, self.shape_list_filename)) as f:
                self.shape_names = f.readlines()[:shapes_per_class]
            self.shape_names = [c+"_"+x.strip() for x in self.shape_names]
            self.shape_names = list(filter(None, self.shape_names))
            temp.append(self.shape_names)

        self.shape_names = [item for sublist in temp for item in sublist]
        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        print('getting information for {} shapes'.format(len(self.shape_names)))
        for shape_ind, shape_name in enumerate(self.shape_names):
            # print('getting information for shape %s' % shape_name)

            def load_pts():
                # load from text file and save in more efficient numpy format
                temp = self.shape_names[shape_ind].split('_')
                c = temp[0]
                id = temp[1]
                point_filename = os.path.join(self.root, c, 'convonet', '43', id, 'pointcloud.npz')
                return np.load(point_filename, mmap_mode=None)['points']

                # point_filename = os.path.join(self.root, c,'04_pts', shape_name + '.xyz')
                # if os.path.isfile(point_filename) or os.path.isfile(point_filename + '.npy'):
                #     pts = file_utils.load_npy_if_valid(point_filename, 'float32', mmap_mode='r')
                #     if pts.shape[1] > 3:
                #         pts = pts[:, 0:3]
                # else:  # if no .xyz file, try .off and discard connectivity
                #     mesh_filename = os.path.join(self.root, '04_pts', shape_name+'.xyz')
                #     pts, _ = mesh_io.load_mesh(mesh_filename)
                #     np.savetxt(fname=os.path.join(self.root, '04_pts', shape_name+'.xyz'), X=pts)
                #     np.save(os.path.join(self.root, '04_pts', shape_name+'.xyz.npy'), pts)
                # return pts

            if self.include_imp_surf:
                if self.reconstruction:
                    # get number of grid points near the point cloud
                    pts = load_pts()
                    # MINE, here is where the query points in inference are decided
                    grid_pts_near_surf_ms = \
                        sdf.get_voxel_centers_grid_smaller_pc(
                            pts=pts, grid_resolution=query_grid_resolution, distance_threshold_vs=self.epsilon)
                    self.shape_patch_count.append(grid_pts_near_surf_ms.shape[0])

                    # un-comment to get a debug output for the necessary query points
                    # mesh_io.write_off('debug/{}'.format(shape_name + '.off'), grid_pts_near_surf_ms, [])
                    # self.shape_patch_count.append(query_grid_resolution ** 3)  # full grid
                else:
                    query_dist_filename = os.path.join(self.root,  shape_name.split('_')[0],'p2s', '05_query_pts', shape_name + '.off.npy')
                    query_dist = np.load(query_dist_filename)
                    self.shape_patch_count.append(query_dist.shape[0])
            else:
                pts = load_pts()

                self.shape_patch_count.append(pts.shape[0])

    # returns a patch centered at the point with the given global index
    # and the ground truth normal at the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, patch_ind = self.shape_index(index)

        def get_patch_points(shape, query_point):

            from source.base import point_cloud

            # optionally always pick the same points for a given patch index (mainly for debugging)
            if self.identical_epochs:
                self.rng.seed((self.seed + index) % (2**32))

            if(self.opt.sensor == "grid"):
                patch_pts_ids = point_cloud.get_patch_kdtree(
                    kdtree=shape.kdtree, rng=self.rng, query_point=query_point,
                    patch_radius=self.patch_radius,
                    points_per_patch=self.points_per_patch*6, n_jobs=self.opt.workers)
                patch_pts_ids = np.random.choice(patch_pts_ids, size=self.points_per_patch, replace=False)
            else:
                patch_pts_ids = point_cloud.get_patch_kdtree(
                    kdtree=shape.kdtree, rng=self.rng, query_point=query_point,
                    patch_radius=self.patch_radius,
                    points_per_patch=self.points_per_patch, n_jobs=self.opt.workers)

            # TODO: here I need to modify the patch when we have auxiliary points, probably easiest to get 1500 NN and choose 300
            # there is a problem with imp_surf_query_point_ms, there are too much when auxiliary points are used, check why

            # find -1 ids for padding
            patch_pts_pad_ids = patch_pts_ids == -1
            patch_pts_ids[patch_pts_pad_ids] = 0
            pts_patch_ms = shape.pts[patch_pts_ids, :]
            # replace padding points with query point so that they appear in the patch origin
            pts_patch_ms[patch_pts_pad_ids, :] = query_point
            patch_radius_ms = utils.get_patch_radii(pts_patch_ms, query_point)\
                if self.patch_radius <= 0.0 else self.patch_radius
            pts_patch_ps = utils.model_space_to_patch_space(
                pts_to_convert_ms=pts_patch_ms, pts_patch_center_ms=query_point,
                patch_radius_ms=patch_radius_ms)

            return patch_pts_ids, pts_patch_ps, pts_patch_ms, patch_radius_ms

        # MINE: they actually load the shape (although from some kind of cache) for each query point again
        shape = self.shape_cache.get(shape_ind)
        imp_surf_query_point_ms = shape.imp_surf_query_point_ms[patch_ind]

        ## MINE: this is where they get the patch from a query point
        # get neighboring points
        patch_pts_ids, patch_pts_ps, pts_patch_ms, patch_radius_ms = \
            get_patch_points(shape=shape, query_point=imp_surf_query_point_ms)
        imp_surf_query_point_ps = utils.model_space_to_patch_space_single_point(
            imp_surf_query_point_ms, imp_surf_query_point_ms, patch_radius_ms)

        patch_sensors_ps = shape.data["sensor_pos"][patch_pts_ids]

        # surf dist can be None because we have no ground truth for evaluation
        # need a number or Pytorch will complain when assembling the batch
        if self.reconstruction:
            imp_surf_dist_ms = np.array([np.inf])
            imp_surf_dist_sign_ms = np.array([np.inf])
        else:
            imp_surf_dist_ms = shape.imp_surf_dist_ms[patch_ind]
            imp_surf_dist_sign_ms = np.sign(imp_surf_dist_ms)
            imp_surf_dist_sign_ms = 0.0 if imp_surf_dist_sign_ms < 0.0 else 1.0

        if self.sub_sample_size > 0:
            pts_sub_sample_ms, ids_sub_sample_ms = utils.get_point_cloud_sub_sample(
                sub_sample_size=self.sub_sample_size, pts_ms=shape.pts,
                query_point_ms=imp_surf_query_point_ms, uniform=self.uniform_subsample)
            sensors_sub_sample_ms = shape.data["sensor_pos"][ids_sub_sample_ms]
        else:
            pts_sub_sample_ms = np.array([], dtype=np.float32)
            sensors_sub_sample_ms = np.array([], dtype=np.float32)



        if not self.reconstruction:
            import trimesh.transformations as trafo
            # random rotation of shape and patch as data augmentation
            rand_rot = trimesh.transformations.random_rotation_matrix(self.rng.rand(3))
            # rand_rot = trimesh.transformations.identity_matrix()
            pts_sub_sample_ms = \
                trafo.transform_points(pts_sub_sample_ms, rand_rot).astype(np.float32)
            patch_pts_ps = \
                trafo.transform_points(patch_pts_ps, rand_rot).astype(np.float32)
            sensors_sub_sample_ms = \
                trafo.transform_points(sensors_sub_sample_ms, rand_rot).astype(np.float32)
            patch_sensors_ps = \
                trafo.transform_points(patch_sensors_ps, rand_rot).astype(np.float32)

            imp_surf_query_point_ms = \
                trafo.transform_points(np.expand_dims(imp_surf_query_point_ms, 0), rand_rot)[0].astype(np.float32)
            imp_surf_query_point_ps = \
                trafo.transform_points(np.expand_dims(imp_surf_query_point_ps, 0), rand_rot)[0].astype(np.float32)
        else:
            a=5

        patch_data = dict()
        # create new arrays to close the memory mapped files

        if(self.opt.sensor=="sensor_vec_norm"):
            patch_sensors_ps = patch_sensors_ps - patch_pts_ps
            patch_sensors_ps = patch_sensors_ps / np.linalg.norm(patch_sensors_ps, axis=1)[:, np.newaxis]
            patch_data['patch_inputs_ps'] = np.concatenate((patch_pts_ps, patch_sensors_ps),axis=1)

            sensors_sub_sample_ms = sensors_sub_sample_ms - pts_sub_sample_ms
            sensors_sub_sample_ms = sensors_sub_sample_ms / np.linalg.norm(sensors_sub_sample_ms, axis=1)[:, np.newaxis]
            patch_data['inputs_sub_sample_ms'] = np.concatenate((pts_sub_sample_ms, sensors_sub_sample_ms), axis=1)
        elif(self.opt.sensor=="grid"):
            patch_sensors_ps = shape.data["sensor_pos"][patch_pts_ids] - patch_pts_ps
            patch_sensors_ps = patch_sensors_ps / np.linalg.norm(patch_sensors_ps, axis=1)[:, np.newaxis]
            patch_data['patch_inputs_ps'] = np.concatenate((patch_pts_ps, patch_sensors_ps, shape.data["ident"][patch_pts_ids]),axis=1)

            sensors_sub_sample_ms = shape.data["sensor_pos"][ids_sub_sample_ms] - pts_sub_sample_ms
            sensors_sub_sample_ms = sensors_sub_sample_ms / np.linalg.norm(sensors_sub_sample_ms, axis=1)[:, np.newaxis]
            patch_data['inputs_sub_sample_ms'] = np.concatenate((pts_sub_sample_ms, sensors_sub_sample_ms, shape.data["ident"][ids_sub_sample_ms]), axis=1)
        else:
            patch_data['patch_inputs_ps'] = patch_pts_ps
            patch_data['inputs_sub_sample_ms'] = pts_sub_sample_ms

        patch_data['patch_radius_ms'] = np.array(patch_radius_ms, dtype=np.float32)
        patch_data['imp_surf_query_point_ms'] = imp_surf_query_point_ms
        patch_data['imp_surf_query_point_ps'] = np.array(imp_surf_query_point_ps)
        patch_data['imp_surf_ms'] = np.array([imp_surf_dist_ms], dtype=np.float32)
        patch_data['imp_surf_magnitude_ms'] = np.array([np.abs(imp_surf_dist_ms)], dtype=np.float32)
        patch_data['imp_surf_dist_sign_ms'] = np.array([imp_surf_dist_sign_ms], dtype=np.float32)

        # un-comment to get a debug output of a training sample
        # import evaluation
        # evaluation.visualize_patch(
        #     patch_pts_ps=patch_data['patch_pts_ps'], patch_pts_ms=pts_patch_ms,
        #     query_point_ps=patch_data['imp_surf_query_point_ps'],
        #     pts_sub_sample_ms=patch_data['pts_sub_sample_ms'], query_point_ms=patch_data['imp_surf_query_point_ms'],
        #     file_path='debug/patch_local_and_global.ply')
        # patch_sphere = trimesh.primitives.Sphere(radius=self.patch_radius, center=imp_surf_query_point_ms)
        # patch_sphere.export(file_obj='debug/patch_sphere.ply')
        # print('Debug patch outputs with radius {} in "debug" dir'.format(self.patch_radius))

        # convert to tensors
        for key in patch_data.keys():
            patch_data[key] = torch.from_numpy(patch_data[key])

        patch_data['shape_ind'] = shape_ind


        return patch_data

    def __len__(self):
        return sum(self.shape_patch_count)

    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        shape_patch_ind = -1
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < (shape_patch_offset + shape_patch_count):
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):

        # point_filename = os.path.join(self.root, '04_pts', self.shape_names[shape_ind]+'.xyz')
        temp = self.shape_names[shape_ind].split('_')
        c = temp[0]
        id = temp[1]
        point_filename = os.path.join(self.root, c, 'convonet', '43', id, 'pointcloud.npz')

        imp_surf_query_filename = os.path.join(self.root,c, 'p2s','05_query_pts', self.shape_names[shape_ind]+'.off.npy') \
            if self.include_imp_surf and self.pre_processed_patches and not self.reconstruction else None
        imp_surf_dist_filename = os.path.join(self.root,c, 'p2s','05_query_dist', self.shape_names[shape_ind]+'.off.npy') \
            if self.include_imp_surf and self.pre_processed_patches and not self.reconstruction else None

        return load_shape(
            point_filename=point_filename,
            imp_surf_query_filename=imp_surf_query_filename,
            imp_surf_dist_filename=imp_surf_dist_filename,
            query_grid_resolution=self.query_grid_resolution,
            epsilon=self.epsilon, sensor=self.opt.sensor
            )
