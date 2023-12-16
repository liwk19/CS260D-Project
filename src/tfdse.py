from config import FLAGS
from saver import saver
from utils import get_root_path, MLP, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, save_pickle, load_pickle
from result import Result
from utils import OurTimer, get_save_path

from os.path import join, basename
from glob import glob, iglob

from math import ceil

from torch_geometric.data import Data, Batch

import networkx as nx
import redis, pickle, random
import numpy as np
from collections import Counter, defaultdict, OrderedDict

from scipy.sparse import hstack

from tqdm import tqdm
from tqdm import trange
import sys

from torch_geometric.data import DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, classification_report, confusion_matrix
from scipy.stats import rankdata, kendalltau
import pandas as pd
from pathlib import Path

import math

KERNEL = FLAGS.tag
db_path = []
for benchmark in FLAGS.benchmarks:
    db_path.append(f'../dse_database/{benchmark}/databases/**/*')


GEXF_FOLDER = join(get_root_path(), 'dse_database', 'programl', '**', 'processed', '**')

# GEXF_FILES = [f for f in sorted(glob(join(GEXF_FOLDER, '*.gexf'))) if f.endswith('.gexf') and KERNEL in f]
TARGETS = ['perf', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF',
           'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil',
                    'nw']
poly_KERNEL = ['atax', 'mvt']
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL

if FLAGS.all_kernels:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf')])
else:
    GEXF_FILES = sorted(
        [f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and KERNEL in f])

def main():
    data_dict, init_feat_dict = get_data()
    model = TFDSENet(init_feat_dict).to(FLAGS.device)
    print(model)
    saver.log_model_architecture(model)


    data_loaders_train = {}
    data_loaders_test = {}
    data_loaders_all = {}
    for gname, data_list in data_dict.items():
        sp = int(len(data_list) * 0.8)
        take_80_perc = data_list[0:sp]
        saver.log_info(f'{gname} Take {len(take_80_perc)} data from {len(data_list)}') # TODO: need shuffle first?
        train_loader = DataLoader(take_80_perc, batch_size=FLAGS.batch_size,
                                  shuffle=False,
                                  pin_memory=True)
        data_loaders_train[gname] = train_loader
        data_loaders_all[gname] = DataLoader(data_list, batch_size=FLAGS.batch_size,
                                  shuffle=False,
                                  pin_memory=True)
        take_20_perc = data_list[sp:]
        data_loaders_test[gname] = DataLoader(take_20_perc, batch_size=FLAGS.batch_size,
                                             shuffle=False,
                                             pin_memory=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    for epoch in trange(FLAGS.epoch_num, file=sys.stdout):

        for gname, train_loader in sorted(data_loaders_train.items()):
            # exit(-1)
            # saver.log_info(f'Epoch {epoch} train: {gname}')
            loss = train(gname, epoch, model, train_loader, optimizer)
            
            if train_losses and loss < min(train_losses):
                if FLAGS.save_model:
                    saver.log_info((f'Saved model at epoch {epoch}'))
                    torch.save(model.state_dict(), join(saver.logdir, "train_model_state_dict.pth"))
            train_losses.append(loss)
            # loss = 0
            # train_acc = test(train_loader)
    saver.log_info(f'min train loss at epoch: {train_losses.index(min(train_losses)) + 1}')
    test(data_loaders_train, f'final_test_train', model, -1, -1)
    test(data_loaders_test, f'final_test_test', model, -1, -1)
    test(data_loaders_all, f'final_test_all',  model, -1, -1)

    print('Done')



def train(gname, epoch, model, train_loader, optimizer):
    model.train()

    total_loss, correct = 0, 0
    i = 0
    for iter, data in enumerate(tqdm(train_loader, position=0, leave=True)):
        data = data.to(FLAGS.device)
        optimizer.zero_grad()

        out, loss = model(data, gname, 'train', epoch, iter)

        # loss = ((out - data.y).pow(2) + 100 * attn_loss).mean()
        loss.backward()
        if FLAGS.task == 'regression':
            total_loss += loss.item()
            saver.writer.add_scalar('loss/loss', loss,
                                epoch * len(train_loader) + i)
            if i % FLAGS.print_every_iter == 0:
                print(f'Iter {i}: Loss {loss}')
        else:
            loss, pred = torch.max(out[FLAGS.target[0]], 1)
            labels = get_y_with_target(data, FLAGS.target[0])
            correct += (pred == labels).sum().item()
            total_loss += labels.size(0)
        optimizer.step()
        
        i += 1

    if FLAGS.task == 'regression':
        return total_loss / len(train_loader)
    else:
        return 1 - correct / total_loss




def test(data_loaders_all, tvt, model, epoch, fold_id):
    model.eval()

    correct, total = 0, 0
    i = 0
    points_dict = OrderedDict()
    input_dict = {}
    input_dict['gname'] = []
    input_dict['key'] = []

    target_list = _get_target_list()
    for target_name in target_list:
        points_dict[target_name] = {'true': [], 'pred': []}

    for gname, loader in sorted(data_loaders_all.items()):
        # saver.log_info(f'Test: {gname}')

        for iter, data in enumerate(tqdm(loader)):
            data = data.to(FLAGS.device)

            out, loss = model(data, gname, tvt, epoch, iter)
            if type(FLAGS.target) is not list:
                out_dict = {}
                out_dict[FLAGS.target] = out
            else:
                out_dict = out
            # pred = out.round().to(torch.long)
            if FLAGS.task == 'regression':
                total += loss.item()
            else:
                loss, pred = torch.max(out[FLAGS.target[0]], 1)
                labels = get_y_with_target(data, FLAGS.target[0])
                correct += (pred == labels).sum().item()
                total += labels.size(0)

            for target_name in target_list:
                if FLAGS.task == 'class':
                    loss, pred = torch.max(out[FLAGS.target[0]], 1)
                    out = pred
                else:
                    out = out_dict[target_name]
                for i in range(len(out)):
                    out_value = out[i].item()
                    pred = out_value
                    true = get_y_with_target(data, target_name)[i].item()
                    points_dict[target_name]['pred'].append(pred)
                    points_dict[target_name]['true'].append(true)

            for i in range(len(out)):
                input_dict['gname'].append(data.gname[i])
                input_dict['key'].append(data.key[i])

            # corrects.append(pred.eq(data.y.to(torch.long)))
            # total_ratio += ratio
            if i % FLAGS.print_every_iter == 0:
                print(f'Iter {i}: Loss {loss}')
            i += 1


    assert (isinstance(FLAGS.target, list))
    saver.save_dict({'input_dict': input_dict, 'points_dict': points_dict},
                    f'fold_{fold_id}/fold_{fold_id}_epoch_{epoch}_{tvt}_pred.pkl')
    
    if FLAGS.task == 'regression':
        result_df = _report_rmse_etc(points_dict,
                                 f'fold {fold_id}_{tvt} epoch {epoch}:')

        return result_df, points_dict, input_dict
    else:
        report_class_loss(points_dict)
        saver.log_info((f'loss is {1 - correct / total}'))
        return 1 - correct / total


target_list_ = None


def _get_target_list():
    global target_list_
    if target_list_ is None:
        _target_list = FLAGS.target
        if not isinstance(FLAGS.target, list):
            _target_list = [FLAGS.target]
        target_list_ = ['actual_perf' if FLAGS.encode_log and t == 'perf' else t
                        for t in
                        _target_list]
    if FLAGS.task == 'class':
        return ['perf']
    return target_list_


def report_class_loss(points_dict):
    d = points_dict[FLAGS.target[0]]
    labels = d['true']
    pred = d['pred']
    target_names = ['invalid', 'valid']
    saver.info('classification report')
    saver.log_info(classification_report(labels, pred, target_names=target_names))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    saver.info(f'Confusion matrix:\n{cm}')
    
    
def _report_rmse_etc(points_dict, label):
    saver.log_info(label)
    data = defaultdict(list)
    tot_mape, tot_rmse, tot_mse, tot_mae, tot_max_err, tot_tau, tot_std = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    num_data = None
    for target_name, d in points_dict.items():
        true_li = d['true']
        pred_li = d['pred']
        if num_data is not None:
            assert num_data == len(true_li)
        num_data = len(true_li)
        assert len(true_li) == len(pred_li)
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

    # data['rmse'].append(f'{tot_rmse:.4f}')
    # data['mse'].append(f'{tot_mse:.4f}')
    # data['tau'].append(f'{tot_tau / len(points_dict):.4f}')
    df = pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    saver.log_info(f'#: {num_data}')
    saver.log_info(df.round(4))
    # exit()
    return df
    # exit()



def get_data():
    # base_csv = pd.read_csv(join(get_root_path(), 'dse_database', 'databases', 'base.csv'))
    # name_cycle_map = dict(zip(base_csv.Kernel_name, base_csv.CYCLE))
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')
    # create a redis databasgtypee
    database = redis.StrictRedis(host='localhost', port=6379)


    data_dict = defaultdict(list)
    init_feat_dict = {}

    all_gs = OrderedDict()

    tot_configs = 0
    num_files = 0

    for gexf_file in tqdm(
            GEXF_FILES[0:]):  # TODO: change for partial/full data
        proceed = False
        for k in ALL_KERNEL:
            if k in gexf_file:
                proceed = True
                break
        if not proceed:
            continue

        database.flushdb()
        g = nx.read_gexf(gexf_file)
        g = nx.convert_node_labels_to_integers(g, ordering='sorted')




        g.variants = OrderedDict()
        gname = basename(gexf_file).split('.')[0]
        g.gname = gname
        saver.log_info(gname)
        all_gs[gname] = g

        n = basename(gexf_file).split('_')[0]
        # db_path = f'./all_dbs/{n}_result.db'
        db_paths = []
        for db_p in db_path:
            
            paths = [f for f in iglob(db_p, recursive=True) if f.endswith(
                '.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' not in f]  # and not 'updated' in f

            db_paths.extend(paths)
        if db_paths is None:
            saver.warning(f'No database found for {n}. Skipping.')
            continue

        db_paths = sorted(db_paths)  # imporant to reduce randomness!

        database.flushdb()
        saver.log_info(f'db_paths for {n}:')
        for d in db_paths:
            saver.log_info(f'{d}')
        if len(db_paths) == 0:
            saver.log_info(f'{n} has no db_paths')

        assert len(db_paths) >= 1

        # load the database and get the keys
        # the key for each entry shows the value of each of the pragmas in the source file
        for idx, file in enumerate(db_paths):
            f_db = open(file, 'rb')
            data = pickle.load(f_db)
            database.hmset(0, data)
            f_db.close()
        # data = pickle.load(f_db)
        # database.hmset(0, mapping=data)
        keys = [k.decode('utf-8') for k in database.hkeys(0)]
        res_reference = 0
        max_perf = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)

            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            if key[0:3] == 'lv1' or obj.perf == 0:  # obj.ret_code.name == 'PASS':
                continue
            if obj.perf > max_perf:
                max_perf = obj.perf
                res_reference = obj

        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            if FLAGS.task == 'regression' and key[0:3] == 'lv1':  # or obj.perf == 0:#obj.ret_code.name == 'PASS':
                continue
            if FLAGS.task == 'regression' and not FLAGS.invalid and obj.perf == 0:
                continue

            x = []
            for _, value in sorted(obj.point.items()):
                if type(value) is str:
                    value = 0
                elif type(value) is int:
                    pass
                else:
                    raise ValueError()
                x.append(value)

            check_dim = init_feat_dict.get(gname)
            if check_dim is not None:
                assert check_dim == len(x)
            else:
                init_feat_dict[gname] = len(x)
            xy_dict = {'x': torch.FloatTensor(np.array([x]))}


            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if FLAGS.norm_method == 'log2':
                            y = math.log2(obj.perf + FLAGS.epsilon)
                        elif 'const' in FLAGS.norm_method:
                            y = obj.perf * FLAGS.normalizer
                            if y == 0:
                                y = FLAGS.max_number * FLAGS.normalizer
                            if FLAGS.norm_method == 'const-log2':
                                y = math.log2(y)
                        elif 'speedup' in FLAGS.norm_method:
                            assert obj.perf != 0
                            #assert got_reference == True
                            y = FLAGS.normalizer / obj.perf
                            if obj.perf == 0:
                                y = 0
                            else:
                                y = res_reference.perf / obj.perf
                            # y = obj.perf / res_reference.perf
                            if FLAGS.norm_method == 'speedup-log2':
                                y = math.log2(y)
                        elif FLAGS.norm_method == 'off':
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                    elif 'util' in tname or 'total' in tname:
                        y = obj.res_util[tname]
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            elif FLAGS.task == 'class':
                if 'lv1' in key:
                    lv2_key = key.replace('lv1', 'lv2')
                    if lv2_key in keys:
                        continue
                    else:
                        y = 0
                else:
                    y = obj.perf if obj.perf == 0 else 1    
                xy_dict['perf'] = torch.FloatTensor(np.array([y])).type(torch.LongTensor)
            else:
                raise NotImplementedError()


            # xy_dict['y'] = y

            # print('@@@@@@@', len(g.variants))
            g.variants[key] = xy_dict

        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')


    saver.log_info(f'Done {num_files} files tot_configs {tot_configs}')



    saver.log_info('Start encoding gs')
    for gname, g in tqdm(all_gs.items()):

        for vname, d_node in g.variants.items():

            if FLAGS.task == 'regression':
                data_dict[gname].append(Data(
                    gname=gname,
                    key=vname,
                    x=d_node['x'],
                    perf=d_node['perf'],
                    actual_perf=d_node['actual_perf'],
                    util_BRAM=d_node['util-BRAM'],
                    util_DSP=d_node['util-DSP'],
                    util_LUT=d_node['util-LUT'],
                    util_FF=d_node['util-FF'],
                    total_BRAM=d_node['total-BRAM'],
                    total_DSP=d_node['total-DSP'],
                    total_LUT=d_node['total-LUT'],
                    total_FF=d_node['total-FF'],
                    xy_dict_programl=d_node,
                ))
            elif FLAGS.task == 'class':
                data_dict[gname].append(Data(
                    x=d_node['x'],
                    perf=d_node['perf'],
                    gname=gname,
                    key=vname
                ))

    # nns = [d.x.shape[0] for d in data_list]
    # print_stats(nns, 'number of nodes')

    for gname, feat_dim in init_feat_dict.items():
        saver.log_info(f'{gname} has initial dim {feat_dim}')

    # saver.log_info(f'dataset[0].num_features {data_list[0].num_features}')
    return data_dict, init_feat_dict






from config import FLAGS
from utils import MLP
import torch
import torch.nn as nn
from collections import OrderedDict


class TFDSENet(torch.nn.Module):
    def __init__(self, init_feat_dict):
        super(TFDSENet, self).__init__()

        D = FLAGS.D

        self.init_MLPs = nn.ModuleDict()

        for gname, feat_dim in init_feat_dict.items():
            mlp = MLP(feat_dim, D,
                             activation_type=FLAGS.activation,
                             num_hidden_lyr=1)
            self.init_MLPs[gname] = mlp

        self.shared_MLPs = MLP(D, D,
                                    activation_type=FLAGS.activation,
                                    num_hidden_lyr=5)


        if FLAGS.task == 'regression':
            self.out_dim = 1
            if FLAGS.loss == 'mse':
                self.loss_fucntion = torch.nn.MSELoss()
            elif FLAGS.loss == 'mape':
                self.loss_fucntion = MAPE()
            elif FLAGS.loss == 'mse_weighted_util':
                self.loss_fucntion = MSE_WEIGHT_UTIL()
            else:
                raise ValueError()
        else:
            self.out_dim = 2
            self.loss_fucntion = torch.nn.CrossEntropyLoss()

        

        self.MLPs = nn.ModuleDict()

        _target_list = FLAGS.target
        if not isinstance(FLAGS.target, list):
            _target_list = [FLAGS.target]
        self.target_list = [t for t in _target_list]
        if FLAGS.task == 'class':
            self.target_list = ['perf']

        for target in self.target_list:
            self.MLPs[target] = MLP(D, self.out_dim,
                                    activation_type=FLAGS.activation,
                                    hidden_channels=[D // 2, D // 4, D // 8],
                                    num_hidden_lyr=3)

    def forward(self, data, gname, tvt='', epoch='', iter=''):
        x = data.x

        MLP_to_use = self.init_MLPs[gname]
        out = MLP_to_use(x)

        out = self.shared_MLPs(out)

        out_dict = OrderedDict()


        total_loss = 0
        out_embed = out
        for target_name in self.target_list:
            out = self.MLPs[target_name](out_embed)
            y = get_y_with_target(data, target_name)
            if FLAGS.task == 'regression':
                target = y.view((len(y), self.out_dim))
                if FLAGS.loss == 'mse_weighted_util':
                    loss = self.loss_fucntion(out, target, target_name=target_name)
                else:
                    loss = self.loss_fucntion(out, target)
            else:
                target = y.view((len(y)))
                loss = self.loss_fucntion(out, target)

            # if FLAGS.loss_scale is not None:
            #     loss = loss * FLAGS.loss_scale[target_name]

            # if FLAGS.margin_loss:
            #     sorted_out = out[torch.argsort(target, dim=0)].view(out.shape)
            #     shifted_delta = (sorted_out - torch.roll(sorted_out, -1, 0))[
            #                     0:-1]
            #     margin_loss = torch.mean(torch.max(
            #         torch.zeros(shifted_delta.shape).to(FLAGS.device),
            #         shifted_delta))
            #     print('margin loss', margin_loss)
            #     total_loss += margin_loss
            out_dict[target_name] = out
            total_loss += loss

        return out_dict, total_loss


class MAPE(torch.nn.Module):
    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            if self.quantiles is None:
                assert y_pred.size(
                    -1) == 1, "Prediction should only have one extra dimension"
                y_pred = y_pred[..., 0]
            else:
                y_pred = y_pred.mean(-1)
        return y_pred

    def forward(self, y_pred, target):
        a = (self.to_prediction(y_pred) - target).abs()
        b = (target.abs() + 1e-8)
        loss = a / b
        rtn = torch.mean(loss)
        return rtn


class MSE_WEIGHT_UTIL(torch.nn.Module):
    def forward(self, y_pred, target, target_name):
        loss = ((y_pred - target) ** 2)
        if 'util' in target_name:
            loss = loss * torch.exp(y_pred - 1)
        rtn = torch.mean(loss)
        return rtn



def get_y_with_target(data, target):
    return getattr(data, target.replace('-', '_'))

main()