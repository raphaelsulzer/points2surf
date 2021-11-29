from __future__ import print_function

import argparse
import os
from datetime import datetime
import random
import math
import shutil
import numbers

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

from source.points_to_surf_model import PointsToSurfModel
from source import points_to_surf_eval
from source import data_loader
from source import sdf_nn
from source.base import evaluation
from source import sdf

import pandas as pd



def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='debug',
                        help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.',
                        help='description')
    parser.add_argument('--indir', type=str, default='datasets/abc_minimal',
                        help='input folder (meshes)')
    parser.add_argument('--outdir', type=str, default='models',
                        help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainset.txt',
                        help='training set file name')
    parser.add_argument('--testset', type=str, default='testset.txt',
                        help='test set file name')
    parser.add_argument('--save_interval', type=int, default='1',
                        help='save model each n epochs')
    parser.add_argument('--debug_interval', type=int, default='1',
                        help='print logging info each n epochs')
    parser.add_argument('--refine', type=str, default='',
                        help='refine model at this path')
    parser.add_argument('--gpu_idx', type=int, default=1,
                        help='set < 0 to use CPU')
    parser.add_argument('--patch_radius', type=float, default=0.05,
                        help='Neighborhood of points that is queried with the network. '
                             'This enables you to set the trade-off between computation time and tolerance for '
                             'sparsely sampled surfaces. Use r <= 0.0 for k-NN queries.')

    # training parameters
    parser.add_argument('--net_size', type=int, default=1024,
                        help='number of neurons in the largest fully connected layer')
    parser.add_argument('--nepoch', type=int, default=101,
                        help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=2,
                        help='input batch size')
    parser.add_argument('--patch_center', type=str, default='point',
                        help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0,
                        help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=100,
                        help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--sub_sample_size', type=int, default=1000,
                        help='number of points of the point cloud that are trained with each patch')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers - 0 means same thread as main execution'
                        'This has to be ZERO (not 1) for debugging. Otherwise debugging doesnt work')
    parser.add_argument('--cache_capacity', type=int, default=1000,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473,
                        help='manual seed')
    parser.add_argument('--single_transformer', type=int, default=0,
                        help='0: two transformers for the global and local information, \n'
                             'rotate local points with matrix trained from global points\n'
                             '1: single transformer for both local and global points')

    parser.add_argument('--uniform_subsample', type=int, default=0,
                        help='1: global sub-sample uniformly sampled from the point cloud\n'
                             '0: distance-depending probability for global sub-sample')
    parser.add_argument('--shared_transformer', type=int, default=0,
                        help='use a single shared QSTN that takes both the local and global point sets as input')
    parser.add_argument('--training_order', type=str, default='random',
                        help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape \n'
                        'remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False,
                        help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--scheduler_steps', type=int, nargs='+', default=[25, 50, 75, 100],
                        help='the lr will be multiplicated with 0.1 at these epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='gradient descent momentum')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean',
                        help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['imp_surf', 'imp_surf_magnitude', 'imp_surf_sign',
                                                                   'patch_pts_ids', 'p_index'],
                        help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'p_index: output debug info to validate the model\n'
                        'imp_surf: distance from query point to patch\n'
                        'imp_surf_magnitude: magnitude for distance from query point to patch\n'
                        'imp_surf_sign: sign for distance from query point to patch\n'
                        'patch_pts_ids: ids for all points in a patch')
    parser.add_argument('--use_point_stn', type=int, default=False,
                        help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True,
                        help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max',
                        help='symmetry operation')
    parser.add_argument('--points_per_patch', type=int, default=300,
                        help='max. number of points per patch')
    parser.add_argument('--debug', type=int, default=0,
                        help='set to 1 of you want debug outputs to validate the model')

    return parser.parse_args(args=args)


def do_logging(writer, log_prefix, epoch, opt, loss, current_step, batchind, num_batch, train, output_names, metrics_dict: dict):
    # current_step = (epoch + fraction_done) * num_batch * opt.batchSize

    loss_cpu = [l.detach().cpu().item() for l in loss]
    loss_sum = sum(loss_cpu)

    writer.add_scalar('loss/{}/total'.format('train' if train else 'eval'), loss_sum, current_step)
    if len(loss_cpu) > 1:
        for wi, w in enumerate(loss_cpu):
            writer.add_scalar('loss/{}/comp_{}'.format('train' if train else 'eval', output_names[wi]),
                              w, current_step)

    keys_to_log = {'abs_dist_rms', 'accuracy', 'precision', 'recall', 'f1_score'}
    for key in metrics_dict.keys():
        if key in keys_to_log and isinstance(metrics_dict[key], numbers.Number):
            value = metrics_dict[key]
            if math.isnan(value):
                value = 0.0
            writer.add_scalar('metrics/{}/{}'.format('train' if train else 'eval', key), value, current_step)

    time = datetime.now()
    time = time.strftime("[%H:%M:%S]")
    state_string = \
        '[{time} Epoch: {epoch}: {batch}/{n_batches}] {prefix} loss: {loss:+.2f}, rmse: {rmse:+.2f}, f1: {f1:+.2f}'.format(
            time=time, name=opt.name, epoch=epoch, batch=batchind, n_batches=num_batch - 1,
            prefix=log_prefix, loss=loss_sum,
            rmse=metrics_dict['abs_dist_rms'], f1=metrics_dict['f1_score'])
    print(state_string)


def points_to_surf_train(opt):

    # TODO: check if patches are really taken from random shape inside one batch, just as sanity check to exclude it as a reason for jumpy loss
    # TODO: add auxiliary points, 3 possible ways:
    # keep 300 nearest neighbors (patch will get smaller)
    # take 1500 nearest neighbors (when adding 4 auxiliary points per point
    # take 300 random points from the 1500 nearest neighbors

    debug = True
    if(debug):
        opt.batchSize = 128
        opt.workers = 0
        n_classes_train = 2
        shapes_per_class_train = 3
        n_classes_test = 1
        shapes_per_class_test = 2
        opt.outdir = "/mnt/raphael/ModelNet10_out/p2s/conventional_debug"
        print_every = 2  # iterations
        val_every = 10  # epochs
        backup_every = 5  # epochs
    else:
        opt.batchSize = 128
        opt.workers = 4
        n_classes_train = 10
        shapes_per_class_train = 1000
        n_classes_test = 10
        shapes_per_class_test = 4
        opt.outdir = "/mnt/raphael/ModelNet10_out/p2s/sensor_grid_300_6"
        print_every = 200  # iterations
        val_every = 8000  # epochs
        backup_every = 10000  # epochs

    # grid_300_6 means sample 1800 nearest neighbors for patch, and randomly select 300 of them

    opt.gpu_idx=2

    # gpu 0 runs with cache_size 1000
    # gpu 1 runs with cache_size 100
    # let's see if it makes a difference,
    # before without sensor (gpu 0) was ~ 10-15mins ahead after 19 epochs

    # opt.input_dim = 6
    # opt.sensor = "sensor_vec_norm"
    opt.input_dim = 8
    opt.sensor = "grid"
    # opt.input_dim = 3
    # opt.sensor = None

    # device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)
    # print('Training on {} GPUs'.format(torch.cuda.device_count()))
    # print('Training on ' + ('cpu' if opt.gpu_idx < 0 else torch.cuda.get_device_name(opt.gpu_idx)))
    device = "cuda:"+str(opt.gpu_idx)

    # colored console output, works e.g. on Ubuntu (WSL)
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'


    os.makedirs(os.path.join(opt.outdir,'model'),exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, "metrics"),exist_ok=True)
    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, 'model', '%s_params.pth' % opt.name)
    model_filename = os.path.join(opt.outdir,'model', '%s_model.pth' % opt.name)
    desc_filename = os.path.join(opt.outdir,'model', '%s_description.txt' % opt.name)

    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        if opt.name != 'test':
            # response = input('A training run named "{}" already exists, overwrite? (y/n) '.format(opt.name))
            response = 'y'
            if response == 'y':
                del_log = True
            else:
                return
        else:
            del_log = True

        if del_log:
            if os.path.exists(log_dirname):
                try:
                    shutil.rmtree(log_dirname)
                except OSError:
                    print("Can't delete " + log_dirname)

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_names = []
    output_loss_weights = dict()
    pred_dim = 0
    for o in opt.outputs:
        if o == 'imp_surf':
            if o not in target_features:
                target_features.append(o)

            output_names.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            output_loss_weights[o] = 1.0
            pred_dim += 1
        elif o == 'imp_surf_magnitude':
            if o not in target_features:
                target_features.append(o)

            output_names.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            output_loss_weights[o] = 1.0  # try higher weight here
            pred_dim += 1
        elif o == 'imp_surf_sign':
            if o not in target_features:
                target_features.append(o)

            output_names.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            output_loss_weights[o] = 1.0
            pred_dim += 1
        elif o == 'p_index':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
        elif o == 'patch_pts_ids':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
        else:
            raise ValueError('Unknown output: %s' % o)

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    # create model
    use_query_point = any([f in opt.outputs for f in ['imp_surf', 'imp_surf_magnitude', 'imp_surf_sign']])
    p2s_model = PointsToSurfModel(
        net_size_max=opt.net_size,
        num_points=opt.points_per_patch,
        output_dim=pred_dim,
        use_point_stn=opt.use_point_stn,
        use_feat_stn=opt.use_feat_stn,
        sym_op=opt.sym_op,
        use_query_point=use_query_point,
        sub_sample_size=opt.sub_sample_size,
        do_augmentation=True,
        single_transformer=opt.single_transformer,
        shared_transformation=opt.shared_transformer,
        input_dim = opt.input_dim,
        sensor=opt.sensor
    )

    start_epoch = 0
    if opt.refine != '':
        print(f'Refining weights from {opt.refine}')
        p2s_model.to(device)
        # p2s_model.cuda(device=device)  # same order as in training
        # p2s_model = torch.nn.DataParallel(p2s_model)
        p2s_model.load_state_dict(torch.load(opt.refine))
        try:
            # expecting a file name like 'vanilla_model_50.pth'
            model_file = str(opt.refine)
            last_underscore_pos = model_file.rfind('_')
            last_dot_pos = model_file.rfind('.')
            start_epoch = int(model_file[last_underscore_pos+1:last_dot_pos]) + 1
            print(f'Continuing training from epoch {start_epoch}')
        except:
            print(f'Warning: {opt.refine} has no epoch in the name. The Tensorboard log will continue at '
                  f'epoch 0 and might be messed up!')

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = data_loader.PointcloudPatchDataset(
        opt=opt,
        root=opt.indir,
        scan="43",
        shape_list_filename=opt.trainset,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        center=opt.patch_center,
        cache_capacity=opt.cache_capacity,
        pre_processed_patches=True,
        sub_sample_size=opt.sub_sample_size,
        num_workers=int(opt.workers),
        patch_radius=opt.patch_radius,
        epsilon=-1,  # not necessary for training
        uniform_subsample=opt.uniform_subsample,
        n_classes=n_classes_train,
        shapes_per_class=shapes_per_class_train
    )
    if opt.training_order == 'random':
        train_datasampler = data_loader.RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = data_loader.SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % opt.training_order)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    ################################################################################
    ################################### TRAIN ######################################
    ################################################################################
    print('Training set: {} patches (in {} batches)'.format(
          len(train_datasampler), len(train_dataloader)))
    opt.train_shapes = train_dataset.shape_names

    log_writer = SummaryWriter(log_dirname, comment=opt.name)
    log_writer.add_scalar('LR', opt.lr, 0)

    # milestones in number of optimizer iterations
    optimizer = optim.SGD(p2s_model.parameters(), lr=opt.lr, momentum=opt.momentum)

    # SGD changes lr depending on training progress
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)  # constant lr
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.scheduler_steps, gamma=0.1)

    if opt.refine == '':
        # p2s_model.cuda(device=device)
        p2s_model.to(device)
        # p2s_model = torch.nn.DataParallel(p2s_model)

    train_num_batch = len(train_dataloader)
    # test_num_batch = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    opt_max=torch.load("models/p2s_max_params.pth")

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    mean_iou = 0.0
    best_iou = 0.0
    best_epoch = 0

    # make an empty results file

    # create the results df
    cols = ['iteration', 'epoch','train_loss_cl', 'train_loss_reg', 'train_loss_total','test_current_iou', 'test_best_iou']
    results_df = pd.DataFrame(columns=cols)
    iterations = 0
    for epoch in range(start_epoch, opt.nepoch, 1):

        for batch_ind, batch_data_train in enumerate(train_dataloader):

            iterations+=1
            row = dict.fromkeys(list(results_df.columns))

            # batch data to GPU
            for key in batch_data_train.keys():
                batch_data_train[key] = batch_data_train[key].to(device)
                # batch_data_train[key] = batch_data_train[key].cuda(non_blocking=True)

            # set to training mode
            p2s_model.train()

            # zero gradients
            optimizer.zero_grad()

            pred_train = p2s_model(batch_data_train)

            loss_train = compute_loss(
                pred=pred_train, batch_data=batch_data_train,
                outputs=opt.outputs,
                output_loss_weights=output_loss_weights,
                fixed_radius=opt.patch_radius > 0.0
            )

            row["iteration"] = iterations
            row["epoch"] = epoch
            row["train_loss_cl"] = loss_train[1].item()
            row["train_loss_reg"] = loss_train[0].item()
            row["train_loss_total"] = loss_train[0].item()+loss_train[1].item()
            row["test_current_iou"] = mean_iou
            row["test_best_iou"] = best_iou

            loss_total = sum(loss_train)

            # back-propagate through entire network to compute gradients of loss w.r.t. parameters
            loss_total.backward()

            # parameter optimization step
            optimizer.step()

            metrics_dict = calc_metrics(outputs=opt.outputs, pred=pred_train, gt_data=batch_data_train)

            # backup model in seperate file

            if(iterations % print_every == 0):

                ### save model
                torch.save(p2s_model.state_dict(), model_filename)

                ### write results to file
                results_df = results_df.append(row, ignore_index=True)
                results_file = os.path.join(opt.outdir,"metrics","results.csv")
                results_df.to_csv(results_file, index=False)

                do_logging(writer=log_writer, log_prefix=green('train'), epoch=epoch, opt=opt, loss=loss_train,
                            current_step=iterations, batchind=batch_ind, num_batch=train_num_batch,
                            train=True, output_names=output_names, metrics_dict=metrics_dict)


            if (iterations % val_every == 0):
                # MINE: reconstruct test shapes using current model
                ### save model
                torch.save(p2s_model.state_dict(), model_filename)

                #################################
                ########### INFERENCE ###########
                #################################
                batch_size = 512  # ~7 GB memory on 1 1070 for 300 patch points + 1000 sub-sample points

                # grid_resolution = 256  # quality like in the paper
                grid_resolution = 128  # quality for a short test
                # this defines the voxel neighborhood around input points for which inference is run. Thus the higher, the more inference points
                # points outside this are classified by the sign propagation
                rec_epsilon = 3

                res_dir_rec = os.path.join(opt.outdir, 'rec')

                print('Points2Surf is reconstructing {} into {}'.format(opt.outdir, res_dir_rec))
                eval_params = [
                    '--gpu_idx', str(opt.gpu_idx),
                    '--indir', opt.indir,
                    '--outdir', opt.outdir,
                    '--dataset', opt.testset,
                    '--query_grid_resolution', str(grid_resolution),
                    '--reconstruction', str(True),
                    '--models', 'vanilla',
                    '--batchSize', str(batch_size),
                    '--workers', str(opt.workers),
                    '--cache_capacity', str(100),
                    '--epsilon', str(rec_epsilon),
                    '--n_classes', str(n_classes_test),
                    '--shapes_per_class', str(shapes_per_class_test)
                ]
                eval_opt = points_to_surf_eval.parse_arguments(eval_params)
                points_to_surf_eval.points_to_surf_eval(eval_opt)

                #################################
                ######### RECONSTRUCTION ########
                #################################
                certainty_threshold = 13
                sigma = 5
                # reconstruct meshes from predicted SDFs
                imp_surf_dist_ms_dir = os.path.join(res_dir_rec, 'dist_ms')
                query_pts_ms_dir = os.path.join(res_dir_rec, 'query_pts_ms')
                vol_out_dir = os.path.join(res_dir_rec, 'vol')
                mesh_out_dir = os.path.join(res_dir_rec, 'mesh')
                mean_iou = sdf.implicit_surface_to_mesh_directory(eval_opt,
                                                                  imp_surf_dist_ms_dir, query_pts_ms_dir,
                                                                  vol_out_dir, mesh_out_dir,
                                                                  grid_resolution, sigma, certainty_threshold,
                                                                  opt.workers)

                if (mean_iou > best_iou):
                    best_iou = mean_iou
                    best_epoch = epoch
                    torch.save(p2s_model.state_dict(), model_filename[:-4] + "_best.pth")

                print("#############################################################")
                print("Mean IoU {} | Best IoU {} at epoch {}".format(mean_iou, best_iou, best_epoch))
                print("#############################################################")

        if(iterations % backup_every):
            # backup metrics file in seperate file
            results_file = os.path.join(opt.outdir, "metrics", "results_" + str(iterations) + ".csv")
            results_df.to_csv(results_file, index=False)

            # backup model in seperate file
            torch.save(p2s_model.state_dict(), model_filename[:-4]+"_"+str(iterations)+".pth")



        # update and log lr
        lr_before_update = scheduler.get_last_lr()
        if isinstance(lr_before_update, list):
            lr_before_update = lr_before_update[0]
        scheduler.step()
        lr_after_update = scheduler.get_last_lr()
        if isinstance(lr_after_update, list):
            lr_after_update = lr_after_update[0]
        if lr_before_update != lr_after_update:
            print('LR changed from {} to {} in epoch {}'.format(lr_before_update, lr_after_update, epoch))
        current_step = (epoch + 1) * train_num_batch * opt.batchSize - 1
        log_writer.add_scalar('LR', lr_after_update, current_step)

        log_writer.flush()





    log_writer.close()


def compute_loss(pred, batch_data, outputs, output_loss_weights, fixed_radius):

    loss = []

    if 'imp_surf' in outputs:
        o_pred = pred.squeeze()
        o_target = batch_data['imp_surf_ms'].squeeze()
        if not fixed_radius:
            o_patch_radius = batch_data['patch_radius_ms']
            o_target /= o_patch_radius
        loss.append(sdf_nn.calc_loss_distance(pred=o_pred, target=o_target) *
                    output_loss_weights['imp_surf'])
    if 'imp_surf_magnitude' in outputs and 'imp_surf_sign' in outputs:
        o_pred = pred[:, 0].squeeze()
        o_target = batch_data['imp_surf_magnitude_ms'].squeeze()
        if not fixed_radius:
            o_patch_radius = batch_data['patch_radius_ms']
            o_target /= o_patch_radius
        loss.append(sdf_nn.calc_loss_magnitude(pred=o_pred, target=o_target) *
                    output_loss_weights['imp_surf_magnitude'])

        o_pred = pred[:, 1].squeeze()
        o_target = batch_data['imp_surf_dist_sign_ms'].squeeze()
        loss.append(sdf_nn.calc_loss_sign(pred=o_pred, target=o_target) *
                    output_loss_weights['imp_surf_sign'])

    return loss


def calc_metrics(outputs, pred, gt_data):

    def compute_rmse_abs_dist(pred, gt):
        abs_dist = sdf_nn.post_process_magnitude(pred)
        rmse = torch.sqrt(torch.mean((abs_dist.abs() - gt.squeeze().abs()) ** 2))
        return rmse.detach().cpu().item()

    def compare_classification(pred, gt):
        inside_class = sdf_nn.post_process_sign(pred)
        eval_dict = evaluation.compare_predictions_binary_tensors(
            ground_truth=gt.squeeze(), predicted=inside_class, prediction_name='training_metrics')
        return eval_dict

    if 'imp_surf_magnitude' in outputs and 'imp_surf_sign' in outputs:
        abs_dist_rms = compute_rmse_abs_dist(pred=pred[:, 0].squeeze(), gt=gt_data['imp_surf_magnitude_ms'])
        eval_dict = compare_classification(pred=pred[:, 1].squeeze(),
                                           gt=gt_data['imp_surf_dist_sign_ms'])
        eval_dict['abs_dist_rms'] = abs_dist_rms
        return eval_dict
    elif 'imp_surf' in outputs:
        abs_dist_rms = compute_rmse_abs_dist(pred=pred.squeeze(), gt=gt_data['imp_surf_ms'])
        pred_class = pred.squeeze()
        pred_class[pred_class < 0.0] = -1.0
        pred_class[pred_class >= 0.0] = 1.0
        eval_dict = compare_classification(pred=pred_class,
                                           gt=gt_data['imp_surf_dist_sign_ms'])
        eval_dict['abs_dist_rms'] = abs_dist_rms
        return eval_dict
    else:
        return {}


if __name__ == '__main__':
    train_opt = parse_arguments()
    points_to_surf_train(train_opt)
