from config import FLAGS
from saver import saver
from utils import get_root_path, MLP, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, load
from result import Result    

from os.path import join, basename
from glob import glob, iglob

from math import ceil

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, Batch

import networkx as nx
import redis, pickle, random
import numpy as np
from collections import Counter, defaultdict, OrderedDict

from scipy.sparse import hstack

from tqdm import tqdm

import os.path as osp

import torch
from torch_geometric.data import Dataset


from shutil import rmtree
import pandas as pd
import math

from data import _coo_to_sparse, create_edge_index, _in_between, \
        _check_any_in_str, _encode_X_torch, _encode_edge_dict, _encode_edge_torch, \
        _encode_X_dict, encode_g_torch, load_encoders, load_all_gs, print_data_stats, _get_y


SAVE_DIR = join(saver.logdir, 'data')

TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']


ENCODER_PATH = join(SAVE_DIR, 'encoders')
# PROCESSED_DIR = join(SAVE_DIR, 'processed')
create_dir_if_not_exists(SAVE_DIR)

DATASET = 'machsuite-poly'
if DATASET == 'cnn1':
    KERNEL = 'cnn'
    db_path = '../dse_database/databases/cnn_case1/'
elif DATASET == 'machsuite':
    KERNEL = FLAGS.tag
    db_path = '../dse_database/machsuite/databases/**/*'
elif DATASET == 'machsuite-poly':
    KERNEL = FLAGS.tag
    db_path = []
    db_path.append('../../yunsheng/software-gnn/src/logs/yunsheng/**/*')
    # for benchmark in FLAGS.benchmarks:
    #     db_path.append(f'../dse_database/{benchmark}/databases/**/*')


GEXF_FOLDER = join(get_root_path(), 'dse_database', 'programl', '**', 'processed', '**')

# TARGETS = ['perf', 'quality', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF',
TARGETS = ['perf', 'util-BRAM', 'util-DSP', 'util-LUT', 'util-FF',
           'total-BRAM', 'total-DSP', 'total-LUT', 'total-FF']
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw']
poly_KERNEL = ['atax', 'mvt', 'gemver', 'gemm-p', '2mm', 'bicg', 'doitgen', 'gesummv', '3mm', 'jacobi-1d', 'fdtd-2d']
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL

if FLAGS.all_kernels:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf')])
else:
    GEXF_FILES = sorted([f for f in iglob(GEXF_FOLDER, recursive=True) if f.endswith('.gexf') and FLAGS.target_kernel in f])



    
class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None):
        # self.processed_dir = PROCESSED_DIR
        super(MyOwnDataset, self).__init__(SAVE_DIR, transform, pre_transform)

    @property
    def raw_file_names(self):
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        # return ['data_1.pt', 'data_2.pt', ...]
        # print(SAVE_DIR)
        rtn = glob(join(SAVE_DIR, '*.pt'))
        return rtn

    def download(self):
        pass

    # Download to `self.raw_dir`.

    def process(self):
        # i = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     data = Data(...)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
        #     i += 1
        pass

    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()

    def get(self, idx):
        data = torch.load(osp.join(SAVE_DIR, 'data_{}.pt'.format(idx)))
        return data



def sample_data_list():
    # f_keys = sorted([f for f in iglob(join(FLAGS.keys_path, '**'), recursive=True) if FLAGS.target_kernel in f and '_19_' in f and '20_design' in f])
    # saver.log_info(f_keys)
    # assert len(f_keys) == 1
    # pickled_keys = pickle.load(open(f_keys[0], 'rb'))
    # SAMPLED_KEYS = [k for name, k in pickled_keys]
    saver.log_info(f'Found {len(GEXF_FILES)} gexf files under {GEXF_FOLDER}')
    # create a redis database
    database = redis.StrictRedis(host='localhost', port=6379)

    ntypes = Counter()
    ptypes = Counter()
    numerics = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()

    assert FLAGS.encoder_path != None
    if FLAGS.encoder_path != None:
        encoders = load(FLAGS.encoder_path)
        enc_ntype = encoders['enc_ntype']
        enc_ptype = encoders['enc_ptype']
        enc_itype = encoders['enc_itype']
        enc_ftype = encoders['enc_ftype']
        enc_btype = encoders['enc_btype']
        
        enc_ftype_edge = encoders['enc_ftype_edge']
        enc_ptype_edge = encoders['enc_ptype_edge']

    data_list = []

    all_gs = OrderedDict()

    X_ntype_all = []
    X_ptype_all = []
    X_itype_all = []
    X_ftype_all = []
    X_btype_all = []
    
    edge_ftype_all = []
    edge_ptype_all = []
    tot_configs = 0
    num_files = 0
    init_feat_dict = {}
    for gexf_file in tqdm(GEXF_FILES[0:]):  
        proceed = False
        if FLAGS.all_kernels:
            for k in ALL_KERNEL:
                if k in gexf_file:
                    proceed = True
                    break
            if not proceed:
                continue
        else:
            assert (len(GEXF_FILES) == 1)


        database.flushdb()
        g = nx.read_gexf(gexf_file)
        g.variants = OrderedDict()
        gname = basename(gexf_file).split('.')[0]
        saver.log_info(gname)
        all_gs[gname] = g

        n = basename(gexf_file).split('_')[0]
        
        db_paths = []
        for db_p in db_path:
            if FLAGS.v_db == 'v20':
                if FLAGS.only_common_db:
                    if FLAGS.test_extra:
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and 'extra' in f] # and not 'updated' in f
                    else:
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and 'common-dse4' in f] # and not 'updated' in f
                else: 
                    if FLAGS.only_new_points:
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith('extra_3.db') and n in f and 'large-size' not in f and not 'archive' in f and 'extra' in f] # and not 'updated' in f
                    else:
                        # paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' in f and 'corrupt' not in f]
                        paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' in f and 'corrupt' not in f \
                            and ('old' not in f) and ('freeze' not in f) and ('yizhou' not in f) and ('-6.db' not in f) and ('only' not in f) and (not f.endswith('d-5.db'))]  # and not 'updated' in f
            else:
                # paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'large-size' not in f and not 'archive' in f and 'v20' not in f and 'single-merged' in f] # and not 'updated' in f
                paths = [f for f in iglob(db_p, recursive=True) if f.endswith('.db') and n in f and 'v20' not in f and 'sampled_20' in f] # and not 'updated' in f
            db_paths.extend(paths)
        if db_paths is None:
            saver.warning(f'No database found for {n}. Skipping.')
            continue

        database.flushdb()
        saver.log_info(f'db_paths for {n}:')
        for d in db_paths:
            saver.log_info(f'{d}')
        if len(db_paths) == 0:
            saver.log_info(f'{n} has no db_paths')

        if FLAGS.v_db == 'v20' and (FLAGS.test_extra or FLAGS.only_new_points):
            if len(db_path) == 0:
                saver.warning(f'no database file for {n}')
                continue
        else:
            assert len(db_paths) >= 1
        
        # load the database and get the keys
        # the key for each entry shows the value of each of the pragmas in the source file
        for idx, file in enumerate(db_paths):
            f_db = open(file, 'rb')
            # print('loading', f_db)
            data = pickle.load(f_db)
            database.hmset(0, data)
            max_idx = idx + 1
            f_db.close()

        keys = [k.decode('utf-8') for k in database.hkeys(0)]
        got_reference = False
        res_reference = 0
        max_perf = 0
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            if key[0:3] == 'lv1' or obj.perf == 0:
                continue
            if obj.perf > max_perf:
                max_perf = obj.perf
                got_reference = True
                res_reference = obj
        if res_reference != 0:
            saver.log_info(f'reference point for {n} is {res_reference.perf}')
        else:
            saver.log_info(f'did not find reference point for {n} with {len(keys)} points')


        # for key in sorted(SAMPLED_KEYS):
        for key in sorted(keys):
            pickle_obj = database.hget(0, key)
            obj = pickle.loads(pickle_obj)
            # try:
            if type(obj) is int or type(obj) is dict:
                continue
            if FLAGS.task == 'regression' and key[0:3] == 'lv1':# or obj.perf == 0:#obj.ret_code.name == 'PASS':
                continue
            if FLAGS.task == 'regression' and not FLAGS.invalid and obj.perf == 0:
                continue
            #### TODO !! fix databases that have this problem:
            if obj.point == {}:
                continue
            # print(key, obj.point)
            xy_dict = _encode_X_dict(
                g, ntypes=ntypes, ptypes=ptypes, itypes=itypes, ftypes=ftypes, btypes = btypes, numerics=numerics, obj=obj)
            edge_dict = _encode_edge_dict(
                g, ftypes=ftypes_edge, ptypes=ptypes_edge)



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
                            y = FLAGS.normalizer / obj.perf
                            # y = res_reference.perf / obj.perf
                            if FLAGS.norm_method == 'speedup-log2':
                                y = math.log2(y)
                        elif FLAGS.norm_method == 'off':
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                        xy_dict['kernel_speedup'] = torch.FloatTensor(np.array([math.log2(res_reference.perf / obj.perf)]))
                        # y = obj.perf
                    elif 'util' in tname or 'total' in tname:
                        y = obj.res_util[tname] * FLAGS.util_normalizer
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

            vname = key

            # print('@@@@@@@', len(g.variants))
            g.variants[vname] = (xy_dict, edge_dict)
            X_ntype_all += xy_dict['X_ntype']
            X_ptype_all += xy_dict['X_ptype']
            X_itype_all += xy_dict['X_itype']
            X_ftype_all += xy_dict['X_ftype']
            X_btype_all += xy_dict['X_btype']
            
            edge_ftype_all += edge_dict['X_ftype']
            edge_ptype_all += edge_dict['X_ptype']
                

        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')
        saver.log_info(f'\tntypes {len(ntypes)}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')


    for gname, g in all_gs.items():
        edge_index = create_edge_index(g)
        saver.log_info('edge_index created', gname)
        for vname, d in g.variants.items():
            d_node, d_edge = d
            X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
            edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)


            if FLAGS.task == 'regression':
                data_list.append(Data(
                    x=X,
                    edge_index=edge_index,
                    # pragmas=d_node['pragmas'],
                    perf=d_node['perf'],
                    actual_perf=d_node['actual_perf'],
                    kernel_speedup=d_node['kernel_speedup'], # base is different per kernel
                    # quality=d_node['quality'],
                    util_BRAM=d_node['util-BRAM'],
                    util_DSP=d_node['util-DSP'],
                    util_LUT=d_node['util-LUT'],
                    util_FF=d_node['util-FF'],
                    total_BRAM=d_node['total-BRAM'],
                    total_DSP=d_node['total-DSP'],
                    total_LUT=d_node['total-LUT'],
                    total_FF=d_node['total-FF'],
                    edge_attr=edge_attr,
                    kernel=gname
                ))
            elif FLAGS.task == 'class':
                data_list.append(Data(
                    x=X,
                    edge_index=edge_index,
                    perf=d_node['perf'],
                    edge_attr=edge_attr,
                    kernel=gname
                ))
            else:
                raise NotImplementedError()


    nns = [d.x.shape[0] for d in data_list]
    print_stats(nns, 'number of nodes')
    ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
    print_stats(ads, 'avg degrees')
    saver.log_info('dataset[0].num_features', data_list[0].num_features)
    for target in TARGETS:
        if not hasattr(data_list[0], target.replace('-', '_')):
            saver.warning(f'Data does not have attribute {target}')
            continue
        ys = [_get_y(d, target).item() for d in data_list]
        # if target == 'quality':
        #     continue
        plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
        saver.log_info(f'{target}_ys', Counter(ys))

    if FLAGS.force_regen:
        saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
        rmtree(SAVE_DIR)
        create_dir_if_not_exists(SAVE_DIR)
        for i in tqdm(range(len(data_list))):
            torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))

    if FLAGS.force_regen:
        from utils import save
        obj = {'enc_ntype': enc_ntype, 'enc_ptype': enc_ptype,
            'enc_itype': enc_itype, 'enc_ftype': enc_ftype,
            'enc_btype': enc_btype, 
            'enc_ftype_edge': enc_ftype_edge, 'enc_ptype_edge': enc_ptype_edge}
        p = ENCODER_PATH
        if FLAGS.encoder_path == None or FLAGS.sample_finetune:
            save(obj, p)
        save(init_feat_dict, join(SAVE_DIR, 'pragma_dim'))
        
        for gname, feat_dim in init_feat_dict.items():
            saver.log_info(f'{gname} has initial dim {feat_dim}')


    rtn = MyOwnDataset()
    return rtn





