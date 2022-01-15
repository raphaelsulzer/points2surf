import os
import time
from source import points_to_surf_eval
from source.base import evaluation
from source import sdf
import pandas as pd
import shutil

# When you see this error:
# 'Expected more than 1 value per channel when training...' which is raised by the BatchNorm1d layer
# for multi-gpu, use a batch size that can't be divided by the number of GPUs
# for single-gpu, use a straight batch size
# see https://github.com/pytorch/pytorch/issues/2584
# see https://forums.fast.ai/t/understanding-code-error-expected-more-than-1-value-per-channel-when-training/9257/12


def full_eval(opt):

    # indir_root = opt.indir
    # outdir_root = os.path.join(opt.outdir, opt.models)

    # indir_root = '/mnt/raphael/ModelNet10'
    # outdir_root = os.path.join('/mnt/raphael/ModelNet10_out/p2s',opt.models)
    # dataset_name = "ModelNet10"


    # indir_root = '/home/qt/sulzerr/scratch/data/ShapeNet'
    # outdir_root = os.path.join('/home/qt/sulzerr/scratch/data/ShapeNet_out/p2s',opt.models)
    # dataset_name = "ShapeNet"
    indir_root = '/mnt/raphael/ShapeNet'
    outdir_root = os.path.join('/mnt/raphael/ShapeNet_out/p2s',opt.models)
    dataset_name = "ShapeNet"

    # if(os.path.exists(os.path.join('/mnt/raphael/ShapeNet_out/p2s',opt.models,'rec'))):
    #     shutil.rmtree(os.path.join('/mnt/raphael/ShapeNet_out/p2s',opt.models,'rec'))

    # indir_root = '/home/qt/sulzerr/scratch/data/ModelNet10'
    # outdir_root = os.path.join('/home/qt/sulzerr/scratch/data/ModelNet10_out/p2s',opt.models)

    opt.n_classes = 100
    opt.shapes_per_class = 1
    opt.classes = ['04530566']
    # opt.classes = None
    opt.scan = 42
    opt.dataset_name = dataset_name
    datasets = opt.dataset
    dataset = "debug.lst"


    print(f'Evaluating on dataset {dataset}')
    opt.indir = os.path.join(indir_root, os.path.dirname(dataset))
    opt.outdir = os.path.join(outdir_root, os.path.dirname(dataset))
    opt.dataset = os.path.basename(dataset)

    # # evaluate
    # if os.path.exists(os.path.join(opt.indir, '05_query_dist')):
    #     opt.reconstruction = False
    #     points_to_surf_eval.points_to_surf_eval(opt)
    #
    #     res_dir_eval = os.path.join(opt.outdir, 'eval')
    #
    #     evaluation.eval_predictions(
    #         os.path.join(res_dir_eval, 'eval'),
    #         os.path.join(opt.indir, '05_query_dist'),
    #         os.path.join(res_dir_eval, 'rme_comp_res.csv'),
    #         unsigned=False)

    # reconstruct
    start = time.time()
    opt.reconstruction = True
    opt.modelpostfix = "_best"
    points_to_surf_eval.points_to_surf_eval(opt)
    res_dir_rec = os.path.join(opt.outdir, 'rec')
    end = time.time()
    print('Inference of SDF took: {}'.format(end - start))
    time_dict = {}
    time_dict["inference"] = end - start

    start = time.time()
    imp_surf_dist_ms_dir = os.path.join(res_dir_rec, 'dist_ms')
    query_pts_ms_dir = os.path.join(res_dir_rec, 'query_pts_ms')
    vol_out_dir = os.path.join(res_dir_rec, 'vol')
    mesh_out_dir = os.path.join(res_dir_rec, 'mesh')
    results_dict = sdf.implicit_surface_to_mesh_directory(opt,
        imp_surf_dist_ms_dir, query_pts_ms_dir,
        vol_out_dir, mesh_out_dir,
        opt.query_grid_resolution,
        opt.sigma,
        opt.certainty_threshold,
        opt.workers)
    end = time.time()
    print('Sign propagation took: {}'.format(end - start))
    time_dict["reconstruction"] = end - start


    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.to_pickle(os.path.join(outdir_root, 'iou_all_' + dataset_name + '.pkl'))

    results_df_class = results_df.groupby(by=['class']).mean()
    results_df_class.loc['mean'] = results_df_class.mean()
    results_df_class.to_csv(os.path.join(outdir_root, 'iou_class_' + dataset_name + '.csv'))


    time_df = pd.DataFrame(time_dict, index=[0])
    time_df.to_csv(os.path.join(outdir_root, 'timing_' + dataset_name + '.csv'))

    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):  # more options can be specified also
        print(results_df)
        print(results_df_class)
        print(time_df)

    a=8

    os.rename(os.path.join('/mnt/raphael/ShapeNet_out/p2s',opt.models,'rec'),os.path.join('/mnt/raphael/ShapeNet_out/p2s',opt.models,'rec_backup'))




if __name__ == '__main__':

    # evaluate model on GT data with multiple query points per patch
    full_eval(opt=points_to_surf_eval.parse_arguments())

    print('MeshNet is finished!')
