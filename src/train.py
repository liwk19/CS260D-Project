from config import FLAGS
from saver import Saver, saver
from utils import MLP, OurTimer, MLP_multi_objective, plot_loss_trend, _get_y_with_target, create_dir_if_not_exists, plot_lr_trend
from data import get_kernel_samples, split_dataset, split_dataset_resample, split_train_test_kernel, MyOwnDataset
import data
SAVE_DIR = data.SAVE_DIR
from model import Net
# from data import print_data_stats
if FLAGS.sample_finetune:
    import sample_finetune
    SAVE_DIR = sample_finetune.SAVE_DIR
import config
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL
poly_KERNEL = config.poly_KERNEL
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL
from coreset import tsne_plot

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix

import torch
import pytorch_warmup as warmup
from torch_geometric.data import DataLoader
import torch.nn as nn
import shutil
import numpy as np

from scipy.stats import rankdata, kendalltau

from tqdm import tqdm
from os.path import join, exists, basename

from collections import OrderedDict, defaultdict

import pandas as pd
import random

import matplotlib.pyplot as plt

def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = [data for data,_ in d['pred']]
    pred = [data for _,data in d['pred']]
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    saver.log_info(classification_report(labels, pred, target_names=target_names))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    saver.info(f'Confusion matrix:\n{cm}')

def _report_rmse_etc(points_dict, label, print_result=True):
    if print_result:
        saver.log_info(label)
    data = defaultdict(list)
    tot_mape, tot_rmse, tot_mse, tot_mae, tot_max_err, tot_tau, tot_std = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    num_data = None
    try:
        for target_name, d in points_dict.items():
            # true_li = d['true']
            # pred_li = d['pred']
            true_li = [data for data,_ in d['pred']]
            pred_li = [data for _,data in d['pred']]
            num_data = len(true_li)
            mape = mean_absolute_percentage_error(true_li, pred_li)
            rmse = mean_squared_error(true_li, pred_li, squared=False)
            mse = mean_squared_error(true_li, pred_li, squared=True)
            mae = mean_absolute_error(true_li, pred_li)
            max_err = max_error(true_li, pred_li)

            true_rank = rankdata(true_li)
            pred_rank = rankdata(pred_li)
            tau = kendalltau(true_rank, pred_rank)[0]
            data['target'].append(target_name)
            data['mape'].append(mape)
            data['rmse'].append(rmse)
            data['mse'].append(mse)
            data['mae'].append(mae)
            data['max_err'].append(max_err)
            data['tau'].append(tau)

            # data['rmse'].append(f'{rmse:.4f}')
            # data['mse'].append(f'{mse:.4f}')
            # data['tau'].append(f'{tau: .4f}')
            tot_mape += mape
            tot_rmse += rmse
            tot_mse += mse
            tot_mae += mae
            tot_max_err += max_err
            tot_tau += tau

            pred_std = d.get('pred_std')
            if pred_std is not None:
                assert type(pred_std) is np.ndarray, f'{type(pred_std)}'
                pred_std = np.mean(pred_std)
                data['pred_std'].append(pred_std)
                tot_std += pred_std
        data['target'].append('tot/avg')
        data['mape'].append(tot_mape)
        data['rmse'].append(tot_rmse)
        data['mse'].append(tot_mse)
        data['mae'].append(tot_mae)
        data['max_err'].append(tot_max_err)
        data['tau'].append(tot_tau / len(points_dict))
        if 'pred_std' in data:
            data['pred_std'].append(tot_std / len(points_dict))
    except ValueError as v:
        saver.log_info(f'Error {v}')
        data = defaultdict(list)

    # data['rmse'].append(f'{tot_rmse:.4f}')
    # data['mse'].append(f'{tot_mse:.4f}')
    # data['tau'].append(f'{tot_tau / len(points_dict):.4f}')
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    if print_result:
        saver.log_info(num_data)
        saver.log_info(df.round(4))
    # exit()
    return df
    # exit()

def feature_extract(model, key_word, gnn_layer=None):
    '''"
        fixes all parameters except for the ones that have "key_word" 
        as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                saver.log_info(f'fixing parameter: {name}')
                param.requires_grad = False
            else:
                if 'conv_first' in name or any([f'conv_layers.{d}' in name for d in range(gnn_layer-1)]):
                    saver.log_info(f'fixing parameter: {name}')
                    param.requires_grad = False
    
    if FLAGS.random_MLP:
        D = FLAGS.D
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]
        if FLAGS.node_attention:
            dim = FLAGS.separate_T + FLAGS.separate_P + FLAGS.separate_pseudo + FLAGS.separate_icmp
            in_D = dim * D
        else:
            in_D = D
        if model.MLP_version == 'single_obj':
            for target in FLAGS.target:
                model.MLPs[target] = MLP(in_D, 1, activation_type=FLAGS.activation,
                                        hidden_channels=hidden_channels,
                                        num_hidden_lyr=len(hidden_channels))
        else:
            model.MLPs = MLP_multi_objective(in_D, 1, activation_type=FLAGS.activation,
                                    hidden_channels=hidden_channels,
                                    objectives=FLAGS.target,
                                    num_common_lyr=FLAGS.MLP_common_lyr)



def check_feature_extract(model, key_word, gnn_layer=None):
    '''"
        checks that all parameters except for the ones that have "key_word" are fixed
        as a result, only "key_word" params will be updated
    '''
    for name, param in model.named_parameters():
        if key_word not in name:
            if not gnn_layer:
                assert param.requires_grad == False
            else:
                if 'conv_first' in name or any([f'conv_layers.{d}' in name for d in range(gnn_layer-1)]):
                    assert param.requires_grad == False


def gen_dataset(li, batch_size=FLAGS.batch_size):
    train_loader = DataLoader(li[0], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    val_loader = DataLoader(li[1], batch_size=batch_size, pin_memory=True, num_workers=4)  # TODO: split make sure no seen kernels in val/test
    test_loader = DataLoader(li[2], batch_size=batch_size, pin_memory=True, num_workers=4)  # TODO

    num_features = train_loader.dataset[0].num_features
    saver.info(f'num features for training: {num_features}')
    edge_dim = train_loader.dataset[0].edge_attr.shape[1]
    saver.info(f'size of the edge attribute is {edge_dim}')
    
    return train_loader, val_loader, test_loader, num_features, edge_dim 

def process_split_data(dataset):
    dataset_dict = defaultdict(list)
    # TODO: only select the kernels used for training
    # new_dataset = []
    # included_kernels, excluded_kernels = [], []
    # for idx, data in tqdm(enumerate(dataset)):
    #     kernel_name = data.kernel.split('_')[0]
    #     if kernel_name in ALL_KERNEL:
    #         new_dataset.append(dataset.get_file_path(idx))
    #         included_kernels.append(kernel_name)
    #     else:
    #         excluded_kernels.append(kernel_name)
    # included_kernels = set(included_kernels)
    # excluded_kernels = set(excluded_kernels)
    # print(len(included_kernels), 'included kernels:', included_kernels)
    # print(len(excluded_kernels), 'excluded kernels:', excluded_kernels)
    # dataset = MyOwnDataset(data_files=new_dataset)

    dataset_dict['train'] = dataset
    dataset_dict['test'] = None
    if not FLAGS.all_kernels:
        dataset = get_kernel_samples(dataset)
        dataset_dict['train'] = dataset
    elif FLAGS.test_kernels is not None:
        dataset_dict = split_train_test_kernel(dataset)
        
    return dataset_dict

def get_train_val_count(num_graphs, val_ratio, test_ratio):
    if FLAGS.sample_finetune:
        r1 = int(num_graphs * 1.0)
        r2 = int(num_graphs * 0)
    elif FLAGS.test_kernels is not None:
        r1 = int(num_graphs * (1.0 - val_ratio))
        r2 = int(num_graphs * (val_ratio))
    else:
        r1 = int(num_graphs * (1.0 - val_ratio - test_ratio))
        r2 = int(num_graphs * (val_ratio))
        
    return r1, r2

def inference(dataset, init_pragma_dict=None, model_path=FLAGS.model_path, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.val_ratio, resample=-1, model_id=0, is_train_set=False, is_val_set=False):
    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)
    if resample == -1:
        li = split_dataset(dataset_dict['train'], r1, r2, dataset_test=dataset_dict['test'])
    else:
        li = split_dataset_resample(dataset, 1.0 - val_ratio - test_ratio, val_ratio, test_ratio, test_id=resample)
    train_loader, val_loader, test_loader, num_features, edge_dim = gen_dataset(li)
    test_set = test_loader 
    if is_train_set: 
        test_set = train_loader
        saver.info('running inference on train set')
    elif is_val_set: 
        test_set = val_loader
        saver.info('running inference on val set')
    
    if init_pragma_dict is None:
        init_pragma_dict = {'all': [1, 21]}
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=init_pragma_dict).to(FLAGS.device)

    if model_path != None:
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        shutil.copy(model_path, join(saver.logdir, f"{(basename(model_path)).split('.')[0]}-{model_id}.pth"))
    else:
        saver.error(f'model path should be set during inference')
        raise RuntimeError()

    if model_id == 0:
        saver.log_model_architecture(model)
    data_list = []
    
    if FLAGS.task == 'regression':
        csv_dict = {'header' : ['gname', 'pragma']}
        test_loss, loss_dict, gae_loss, MSE_loss = test(test_set, 'test', model, 0, plot_test = True, csv_dict = csv_dict, data_list =data_list, is_train_set=is_train_set, is_val_set=is_val_set)
        loss_dict = {k: round(v, 4) for k, v in loss_dict.items()}
        saver.log_info((f'{loss_dict}'))
        saver.log_info(('Test loss: {:.7f}, MSE loss: {:.7f}'.format(test_loss, MSE_loss)))
        saver.log_dict_of_dicts_to_csv(f'actual-prediction-{model_id}', csv_dict, csv_dict['header'])
        print(len(data_list), 'out of', len(test_loader))
        if FLAGS.get_forgetting_examples:
            NEW_SAVE_DIR = SAVE_DIR.replace('round', 'forgetting-round')
            saver.log_info(f'Saving {len(data_list)} to disk {NEW_SAVE_DIR}; Deleting existing files')
            # if exists(NEW_SAVE_DIR):
            #     rmtree(NEW_SAVE_DIR)
            # create_dir_if_not_exists(NEW_SAVE_DIR)
            # for i in tqdm(range(len(data_list))):
            #     torch.save(data_list[i], join(NEW_SAVE_DIR, 'data_{}.pt'.format(i)))
    else:
        test_loss, loss_dict_test = test(test_loader, 'test', model, 0)
        saver.log_info(('Test loss: {:.3f}'.format(test_loss)))
        

def model_update(model, losses_list, loss, epoch, plot_test, tag, saver=saver):
    saver.writer.add_scalar(f'{tag}/{tag}', loss, epoch)
    if losses_list and loss < min(losses_list):
        if FLAGS.save_model:
            saver.log_info((f'Saved {tag} model at epoch {epoch}'))
            # TODO: write save_epoch
            save_epoch = (int(epoch / 100) + 1) * 100
            torch.save(model.state_dict(), join(saver.model_logdir, f"{save_epoch}_{tag}_model_state_dict.pth"))
            print(join(saver.model_logdir, f"{save_epoch}_{tag}_model_state_dict.pth"))
        plot_test = True
    losses_list.append(loss)
        
    return plot_test

def log_loss(loss_dict, gae_loss, tag, saver):
    saver.log_info((f'{tag} GAE loss: {gae_loss}'))
    saver.log_info((f'{tag} loss breakdown {loss_dict}'))


# TODO: added the ensemble inference method here
def ensemble_inference(dataset, pragma_dim = None, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.val_ratio, resample=-1):
    saver.info(f'Reading dataset from {SAVE_DIR}')    
    
    num_features = dataset[0].num_features
    edge_dim = dataset[0].edge_attr.shape[1]

    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)
    
    #TODO: use uncertainty estimation method to find coreset
    if FLAGS.uncertainty_metric != 'False':
        li, raw_li = split_dataset(dataset_dict['train'], FLAGS.transfer_k_shot, r2,
                    dataset_test=dataset_dict['test'], pragma_dim=pragma_dim,
                    num_features=dataset[0].num_features,
                    edge_dim=dataset[0].edge_attr.shape[1])
    else:
        li, raw_li = split_dataset(dataset_dict['train'], FLAGS.transfer_k_shot, r2,
                            dataset_test=dataset_dict['test'], pragma_dim=pragma_dim)
    if FLAGS.tsne_plot:
        tsne_plot(raw_li, epoch=0, pragma_dim=pragma_dim)
    train_loader = DataLoader(li[0], batch_size=len(li[0])+10, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(li[2], batch_size=len(li[2])+10, pin_memory=True, num_workers=4)
    assert len(train_loader) == len(test_loader) == 1
    ensemble_train_loss, ensemble_test_loss = [], []
    ensemble_test_out = []

    for model_idx in range(len(FLAGS.ensemble_KERNEL)):
        model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
        model_path = f'{FLAGS.ensemble_model_path}_{ALL_KERNEL[0]}_{FLAGS.ensemble_KERNEL[model_idx]}'
        model_path += f'_{FLAGS.uncertainty_metric}/run1/{FLAGS.epoch_num}_train_model_state_dict.pth'
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        if FLAGS.feature_extract:
            feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
        model.eval()
        with torch.no_grad():
            for data in train_loader:
                data = data.to(FLAGS.device)
                train_out, train_loss, _, _ = model.to(FLAGS.device)(data)
            for data in test_loader:
                data = data.to(FLAGS.device)
                test_out, test_loss, _, _ = model.to(FLAGS.device)(data)

        ensemble_train_loss.append(train_loss.item())
        ensemble_test_loss.append(test_loss.item())
        ensemble_test_out.append(test_out)
    
    print(f'Mean train loss: {np.mean(ensemble_train_loss):.3f}+-{np.std(ensemble_train_loss):.3f}')
    print(f'Mean test loss: {np.mean(ensemble_test_loss):.3f}+-{np.std(ensemble_test_loss):.3f}')
    
    if FLAGS.ensemble_metric == 'lowest_loss':
        idx = np.argmin(ensemble_train_loss)
        print(FLAGS.ensemble_KERNEL[idx])
        print(f'Test loss: {ensemble_test_loss[idx]:.3f}')
    else:
        assert FLAGS.ensemble_metric in ['average', 'weighted', 'similarity_euclidean', 'similarity_cosine']
        summary_test_out = {}
        weights = []
        for idx, test_out in enumerate(ensemble_test_out):
            if FLAGS.ensemble_metric == 'weighted':
                weights.append(1 / ensemble_train_loss[idx])
            elif FLAGS.ensemble_metric in ['similarity_euclidean', 'similarity_cosine']:
                hidden_path = '/home/weikaili/software-gnn/src/hidden_save' 
                test_hidden = torch.from_numpy(np.load(f'{hidden_path}/{poly_KERNEL[0]}.npy')) # N x 128
                ensemble_idx_hdden = torch.from_numpy(np.load(f'{hidden_path}/{FLAGS.ensemble_KERNEL[idx]}.npy')) # M x 128
                test_hidden =  torch.mean(test_hidden, dim=0) # 128
                ensemble_idx_hdden = torch.mean(ensemble_idx_hdden, dim=0) # 128
                if FLAGS.ensemble_metric == 'similarity_euclidean':
                    dist = (test_hidden - ensemble_idx_hdden).pow(2).sum(dim=0)
                    weights.append(1/dist)
                elif FLAGS.ensemble_metric == 'similarity_cosine':
                    dist = torch.nn.functional.cosine_similarity(test_hidden, ensemble_idx_hdden, dim=0)
                    weights.append(dist)
            else:
                weights.append(1)
            for k, v in test_out.items():
                if k not in summary_test_out.keys():
                    summary_test_out[k] = [v]
                else:
                    summary_test_out[k].append(v)
        print(weights)
        for k, v in summary_test_out.items():
            v = torch.stack(v)
            if FLAGS.ensemble_metric in ['weighted', 'similarity_euclidean', 'similarity_cosine'] :
                sum_v = v[0] * weights[0]
                for i in range(1, len(v)):
                    sum_v += v[i] * weights[i]
                summary_test_out[k] = sum_v / np.sum(weights)
            else:
                summary_test_out[k].append(v)
    
    for data in test_loader:
        data = data.to(FLAGS.device)
        test_loss = model.to(FLAGS.device).calculate_total_loss(data, summary_test_out)
    # print(f'Train loss: {train_loss.item():.3f}')
    print(f'Test loss: {test_loss.item():.3f}')


def train_main(dataset, pragma_dim = None, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.val_ratio, resample=-1, e_kernel=None):
    global saver
    if e_kernel is not None:
        saver = Saver(e_kernel)
    saver.info(f'Reading dataset from {SAVE_DIR}')
    
    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict['train'])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)

    # dataset_dict['train'] contains all the training data
    # visualize_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(dataset_dict['train'])

    # TODO: I added this branch
    if FLAGS.finetune:
        assert dataset_dict['test'] == None
        if FLAGS.coreset == 'CRAIG':
            li, weight, raw_li = split_dataset(dataset_dict['train'], FLAGS.transfer_k_shot, r2,
                                       dataset_test=dataset_dict['test'], pragma_dim=pragma_dim)
            if FLAGS.tsne_plot:
                tsne_plot(raw_li, epoch=0, pragma_dim=pragma_dim)
        elif FLAGS.coreset == 'ensemble_train':
            li, raw_li = split_dataset(dataset_dict['train'], FLAGS.transfer_k_shot, r2,
                        dataset_test=dataset_dict['test'], pragma_dim=pragma_dim,
                        num_features=dataset[0].num_features,
                        edge_dim=dataset[0].edge_attr.shape[1])
            li[2] = []
            if FLAGS.tsne_plot:
                tsne_plot(raw_li, epoch=0, pragma_dim=pragma_dim)
        elif FLAGS.coreset == 'pretrain_test':
            file_li = dataset.processed_file_names
            dataset = MyOwnDataset(data_files=file_li)
            li = [dataset, [], []]
        else:
            li, raw_li = split_dataset(dataset_dict['train'], FLAGS.transfer_k_shot, r2,
                                       dataset_test=dataset_dict['test'], pragma_dim=pragma_dim)
    elif resample == -1:
        li = split_dataset(dataset_dict['train'], r1, r2, dataset_test=dataset_dict['test'])
    else:
        li = split_dataset_resample(dataset_dict['train'], 1.0 - val_ratio - test_ratio, val_ratio, test_ratio, test_id=resample)
    train_loader, val_loader, test_loader, num_features, edge_dim = gen_dataset(li)
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)

    if e_kernel != None:
        model_path = f'{FLAGS.ensemble_model_path}_{e_kernel}/run1/'
        model_path += f'{FLAGS.ensemble_model_epoch}_train_model_state_dict.pth'
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    elif FLAGS.model_path != None:
        model_path = FLAGS.model_path[0] if type(FLAGS.model_path) is list else FLAGS.model_path 
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
    # for datalist in test_loader:
    #     for i in range(len(datalist)):
    #         saver.log_info(datalist[i].kernel)

    if FLAGS.feature_extract:
        feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)

    saver.log_model_architecture(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    
    num_steps = len(train_loader) * FLAGS.epoch_num
    
    if FLAGS.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[FLAGS.epoch_num // 3], gamma=0.1)
    elif FLAGS.scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    if FLAGS.warmup == 'linear':
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif FLAGS.warmup == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif FLAGS.warmup == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    else:
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)

    train_losses, val_losses, test_losses, total_lrs = [], [], [], []
    gae_train_losses, gae_val_losses, gae_test_losses = [], [], []
    plot_test = False
    if FLAGS.finetune and FLAGS.coreset == 'pretrain_test':
        assert len(ALL_KERNEL) == 1
        hidden_list = []
        with torch.no_grad():
            for data in tqdm(train_loader):
                data = data.to(FLAGS.device)
                _, _, _, _, hidden = model.to(FLAGS.device)(data, return_middle=7)
                hidden_list.append(hidden.cpu().numpy())
        hidden_list = np.concatenate(hidden_list, 0)
        # hidden_list = np.mean(hidden_list, 0)
        np.save(f'hidden_save/{ALL_KERNEL[0]}.npy', hidden_list)
        exit()
    if len(test_loader) > 0:
        test_loss, loss_dict_test, gae_loss_test, _ = test(test_loader, 'test', model, 0, plot_test, test_losses)
        print(f'initial test loss: {test_loss:.4f}')
    
    # TODO: use the test result at the best epoch
    best_val, final_test = 100, 100
    for epoch in range(FLAGS.epoch_num):
        plot_test = False
        timer = OurTimer()
        if FLAGS.feature_extract:
            check_feature_extract(model, 'MLPs', FLAGS.fix_gnn_layer)
        saver.log_info(f'Test batch ID (resample): {resample} - Epoch {epoch} train')
        # TODO: I added the parameter "use_MAML"
        if FLAGS.coreset == 'CRAIG' and FLAGS.finetune == True:
            loss, loss_dict_train, gae_loss_train, lrs = train(epoch, model, train_loader, optimizer, lr_scheduler, warmup_scheduler, weight)
        else:
            loss, loss_dict_train, gae_loss_train, lrs = train(epoch, model, train_loader, optimizer, lr_scheduler, warmup_scheduler)
        plot_test = model_update(model, train_losses, loss, epoch, plot_test, 'train', saver)
        total_lrs.extend(lrs)
        
        if epoch % 10 == 9:
            if len(val_loader) > 0:
                saver.log_info(f'Epoch {epoch} val')
                val, loss_dict_val, gae_loss_val, _ = test(val_loader, 'val', model, epoch)
                plot_test = model_update(model, val_losses, val, epoch, plot_test, 'val', saver)
            if len(test_loader) > 0:
                saver.log_info(f'Epoch {epoch} test')
                test_loss, loss_dict_test, gae_loss_test, _ = test(test_loader, 'test', model, epoch, plot_test, test_losses)
                plot_test = model_update(model, test_losses, test_loss, epoch, plot_test, 'test', saver)
            if len(val_loader) > 0 and len(test_loader) > 0:
                if val < best_val:
                    best_val = val
                    final_test = test_loss

            log_loss(loss_dict_train, gae_loss_train, "Train", saver)
            if len(val_loader) > 0 and len(test_loader) > 0:
                log_loss(loss_dict_val, gae_loss_val, "Val", saver)
                log_loss(loss_dict_test, gae_loss_test, "Test", saver)
                saver.log_info(('Epoch: {:03d}, Train Loss: {:.4f}, Val loss: {:.4f}, '
                            'Test: {:.4f}) Time: {}'.format(
                            epoch, loss, val, test_loss, timer.time_and_clear())))
                gae_val_losses.append(gae_loss_val)
                gae_test_losses.append(gae_loss_test)
            elif len(test_loader) > 0:
                log_loss(loss_dict_test, gae_loss_test, "Test", saver)
                saver.log_info(('Epoch: {:03d}, Train loss: {:.4f}, '
                                'Test: {:.4f}) Time: {}'.format(
                                epoch, loss, test_loss, timer.time_and_clear())))
                gae_test_losses.append(gae_loss_test)
            else:
                saver.log_info(('Epoch: {:03d}, Train loss: {:.4f}, '
                                'Time: {}'.format(
                    epoch, loss, timer.time_and_clear())))
            gae_train_losses.append(gae_loss_train)
            
            if len(train_losses) > 50 and len(test_loader) > 0:
                if len(set(train_losses[-50:])) == 1 and len(set(test_losses[-50:])) == 1:
                    break
            
            # TODO: visualize
            if epoch % 25 == 24 and FLAGS.finetune:
                if FLAGS.tsne_plot:
                    tsne_plot(raw_li, epoch=epoch, pragma_dim=pragma_dim, model=model)
    
    if e_kernel is not None:
        return

    print(f'Best val: {best_val:.4f}, test: {final_test:.4f}; last test: {test_loss:.4f}')
    if FLAGS.finetune == True:
        if FLAGS.coreset == 'random':
            np.save(f'val_losses/{poly_KERNEL[0]}_random_{FLAGS.transfer_k_shot}_{FLAGS.transfer_save_num}.npy', val_losses)
            np.save(f'test_losses/{poly_KERNEL[0]}_random_{FLAGS.transfer_k_shot}_{FLAGS.transfer_save_num}.npy', test_losses)
        else:
            np.save(f'val_losses/{poly_KERNEL[0]}_{FLAGS.coreset}_{FLAGS.craig_metric}_{FLAGS.transfer_k_shot}_{FLAGS.transfer_save_num}.npy', val_losses)
            np.save(f'test_losses/{poly_KERNEL[0]}_{FLAGS.coreset}_{FLAGS.craig_metric}_{FLAGS.transfer_k_shot}_{FLAGS.transfer_save_num}.npy', test_losses)
    exit()
    epochs = range(epoch+1)
    plot_loss_trend(epochs, train_losses, val_losses, test_losses, saver.get_log_dir(), file_name='losses.png')
    if FLAGS.gae_T or FLAGS.gae_P:
        plot_loss_trend(epochs, gae_train_losses, gae_val_losses, gae_test_losses, saver.get_log_dir(), file_name='gae_losses.png')
    if len(test_loader) > 0:
        saver.log_info(f'min test loss at epoch: {test_losses.index(min(test_losses))}')
    if len(val_loader) > 0:
        saver.log_info(f'min val loss at epoch: {val_losses.index(min(val_losses))}')
    if FLAGS.scheduler is not None:
        plot_lr_trend(total_lrs, FLAGS.epoch_num + 1, saver.get_log_dir())
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses))}')
    
def set_target_list():    
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if FLAGS.task =='regression':
        target_list = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t for t in _target_list]
    else:
        target_list = [_target_list[0]]
    
    loss_dict = {}
    for t in target_list:
        loss_dict[t] = 0.0
        
    return target_list, loss_dict

def update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total_loss, correct):
    if FLAGS.task == 'regression':
        total_loss += loss.item() # * data.num_graphs
        if not FLAGS.SSL:
            for t in target_list:
                loss_dict[t] += loss_dict_[t].item()
        return loss_dict, total_loss
    else:
        loss_, pred = torch.max(out_dict[FLAGS.target[0]], 1)
        labels = _get_y_with_target(data, FLAGS.target[0])
        correct += (pred == labels).sum().item()
        total_loss += labels.size(0)
        return pred, correct, total_loss

def train(epoch, model, train_loader, optimizer, lr_scheduler, warmup_scheduler, weight=None):
    model.train()
    lrs = []
    total_loss, correct, i = 0, 0, 0
    target_list, loss_dict = set_target_list()
    for data in tqdm(train_loader):
        if FLAGS.scheduler is not None:
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            if i == 0:
                saver.log_info(f"epoch = {epoch}, learning rate = {lr}")
        data = data.to(FLAGS.device)

        if FLAGS.use_MAML == False:
            out_dict, loss, loss_dict_, gae_loss = model.to(FLAGS.device)(data, weight=weight)
        else:  # Still has problems unsolved. Don't use it now.
            index_list = list(range(len(data.kernel)))
            random.shuffle(index_list)
            valid_indices = index_list[:FLAGS.transfer_k_shot]
            tmp_model = copy.deepcopy(model)
            tmp_optimizer = torch.optim.Adam(tmp_model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
            out_dict, loss, loss_dict_, gae_loss = tmp_model.to(FLAGS.device)(data, valid_indices)
            tmp_optimizer.zero_grad()
            loss.backward()
            tmp_optimizer.step()
            valid_indices = index_list[FLAGS.transfer_k_shot:]
            out_dict, loss, loss_dict_, gae_loss = tmp_model.to(FLAGS.device)(data, valid_indices)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if FLAGS.scheduler is not None:
            lr_scheduler.step(lr_scheduler.last_epoch+1)
        if FLAGS.warmup is not None:
            warmup_scheduler.dampen()
        
        total_loss_dict = update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total_loss, correct)
        if FLAGS.task == 'regression': loss_dict, total_loss = total_loss_dict
        else: pred, correct, total_loss = total_loss_dict
        
        saver.writer.add_scalar('loss/loss', loss, epoch * len(train_loader) + i)
        if 'GNLL' in FLAGS.loss:
            for target in target_list:
                out = out_dict[target]
                sigma = out[:, 1].reshape(-1, 1)
                mu = out[:, 0].reshape(-1, 1)
                saver.writer.add_scalar(f'train/sigma-{target}', torch.mean(torch.sqrt(torch.exp(sigma))), epoch * len(train_loader) + i)
                saver.writer.add_scalar(f'train/mu-{target}', torch.mean(mu), epoch * len(train_loader) + i)
        i += 1
    
    if FLAGS.scheduler is not None and epoch < 2:
        create_dir_if_not_exists(join(saver.get_log_dir(), 'lrs'))
        # plot_lr_trend(lrs, epoch, join(saver.get_log_dir(), 'lrs'))
    if FLAGS.task == 'regression':
        return total_loss / len(train_loader), {key: v / len(train_loader) for key, v in loss_dict.items()}, gae_loss, lrs
    else:
        return 1 - correct / total_loss, {key: v / len(train_loader) for key, v in loss_dict.items()}, gae_loss, lrs


def inference_loss_function(pred, true):
    return (pred - true) ** 2

def update_csv_dict(csv_dict, data, i, target_name, target_value, out_value):
    if csv_dict is not None:
        gname = _get_y_with_target(data, 'gname')[i]
        pragma = _get_y_with_target(data, 'pragmas')[i][0].item()
        pragma = '-'.join([str(j.item()) for j in _get_y_with_target(data, 'pragmas')[i]])
        if True or 'blocked' in gname:
            if f'{gname}-{pragma}' not in csv_dict:
                csv_dict[f'{gname}-{pragma}'] = {'gname': gname, 'pragma': pragma}
            csv_dict[f'{gname}-{pragma}'][f'acutal-{target_name}'] = target_value
            csv_dict[f'{gname}-{pragma}'][f'predicted-{target_name}'] = out_value
            l = csv_dict['header']
            if f'acutal-{target_name}' not in l:
                l.extend([f'acutal-{target_name}', f'predicted-{target_name}'])
                csv_dict['header'] = l

def test(loader, tvt, model, epoch, plot_test = False, test_losses = [-1], csv_dict = None, data_list = [], is_train_set=False, is_val_set=False):
    model.eval()
    my_softplus = nn.Softplus()
    inference_loss, correct, total, count_data = 0, 0, 0, 1
    points_dict = OrderedDict()
    target_list, loss_dict = set_target_list()
    for target_name in target_list:
        points_dict[target_name] = {'true': [], 'pred': [], 'sigma_mu': [], 'sigma+mu': [], 'sigma':[], 'error': []}
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(FLAGS.device)
            out_dict, loss, loss_dict_, gae_loss = model.to(FLAGS.device)(data)
            total_loss_dict = update_total_loss(loss, data, target_list, loss_dict, loss_dict_, out_dict, total, correct)
            if FLAGS.task == 'regression': loss_dict, total = total_loss_dict
            else: pred, correct, total = total_loss_dict  

            if not FLAGS.SSL:
                for target_name in target_list:
                    if 'inf' in FLAGS.subtask:
                        saver.info(f'{target_name}')
                    if FLAGS.task == 'class': out = pred
                    elif FLAGS.encode_log and 'perf' in target_name: out = out_dict['perf'] 
                    else: out = out_dict[target_name]
                        
                    if 'GNLL' in FLAGS.loss:
                        if FLAGS.loss == 'myGNLL':
                            sigma = torch.sqrt(torch.exp(out[:, 1].reshape(-1, 1)))
                        else:
                            sigma = torch.sqrt(my_softplus(out[:, 1].reshape(-1, 1)))
                        out_ = out
                        out = out[:, 0].reshape(-1, 1)
                    for i in range(len(out)):
                        out_value = out[i].item()
                        target_value = _get_y_with_target(data, target_name)[i].item()
                        if FLAGS.encode_log and target_name == 'actual_perf':
                            out_value = 2**(out_value) * (1 / FLAGS.normalizer)
                        if 'inf' in FLAGS.subtask:
                            inference_loss += inference_loss_function(out_value, target_value)
                            count_data += 1
                            update_csv_dict(csv_dict, data, i, target_name, target_value, out_value)
                                        
                            if FLAGS.get_forgetting_examples:
                                diff_pred = abs(out_value - target_value)
                                add = True if ('perf' in target_name and diff_pred > 0.5) or ('util' in target_name and diff_pred > 0.3) else False
                                if add:
                                    data_list.append(data)
                                    pragma_configs = [p if type(p) is str else str(p.item()) for p in _get_y_with_target(data, 'pragmas')[i]]
                                    saver.info(f"data {i} {_get_y_with_target(data, 'gname')[i]} pramga {'-'.join(pragma_configs)} actual value: {target_value:.2f}, predicted value: {out_value:.2f}")
                            elif out_value != target_value: # and sigma[i].item() > 0.57:
                                saver.info(f"{target_name} data {i} {_get_y_with_target(data, 'gname')[i]} pramga {_get_y_with_target(data, 'pragmas')[i][0].item()} actual value: {target_value:.2f}, predicted value: {out_value:.2f}") #, sigma: {sigma[i].item()}, log_var: {out_[i, 1].item()}')")
                            
                        points_dict[target_name]['pred'].append((target_value, out_value))
                        points_dict[target_name]['true'].append((target_value, target_value))
                        points_dict[target_name]['error'].append((target_value, abs(target_value - out_value)))
                        
                        if 'GNLL' in FLAGS.loss: # and FLAGS.subtask != 'inference':
                            points_dict[target_name]['sigma_mu'].append((target_value, out[i].item() - sigma[i].item()))
                            points_dict[target_name]['sigma'].append((target_value, sigma[i].item()))
                            points_dict[target_name]['sigma+mu'].append((target_value, out[i].item() + sigma[i].item()))

    if FLAGS.task != 'class' and FLAGS.plot_pred_points and tvt == 'test' and (plot_test or (test_losses and (total / len(loader)) < min(test_losses))):
        from utils import plot_points_with_subplot, plot_points_with_subplot_sigma
        saver.log_info(f'@@@ plot_pred_points')
        assert(isinstance(FLAGS.target, list))
        use_sigma = True if 'GNLL' in FLAGS.loss else False
        label = f'epoch_{epoch+1}_{tvt}_train' if is_train_set else f'epoch_{epoch+1}_{tvt}_test'
        if is_val_set: label = f'epoch_{epoch+1}_{tvt}_val'
        if 'inf' in FLAGS.subtask or 'GNLL' not in FLAGS.loss:
            plot_points_with_subplot(points_dict, label, saver.plotdir, target_list, use_sigma=use_sigma)
        if 'GNLL' in FLAGS.loss:
            plot_points_with_subplot_sigma(points_dict, label, saver.plotdir, target_list, use_sigma=use_sigma)
            

    if FLAGS.task == 'regression':
        if 'inf' in FLAGS.subtask:
            _report_rmse_etc(points_dict, f'epoch {epoch}:', True)
        return (total / len(loader), {key: v / len(loader) for key, v in loss_dict.items()}, gae_loss, inference_loss / count_data * len(target_list))
    else:
        if 'inf' in FLAGS.subtask: report_class_loss(points_dict)
        return 1 - correct / total, {key: v / len(loader) for key, v in loss_dict.items()}, gae_loss, 0


