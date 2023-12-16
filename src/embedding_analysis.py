from model import Net
from saver import saver
from config import FLAGS
from utils import _get_y_with_target, create_dir_if_not_exists, print_stats

import torch
from torch_geometric.data import DataLoader
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

from os.path import basename, join
import shutil
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import numpy as np

TARGETS = ['actual_perf', 'perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
if FLAGS.separate_T:
    TargetEmb = ['emb_P', 'emb_T', 'whole', 'pragmas']
else:
    TargetEmb = ['emb_P', 'pragmas']
######
# if mode = 'tsne' --> design_points = {kernel_name: [[emb_P, emb_T, TARGETS]]}
######

all_count = 0
def get_model(dataset, model_path=FLAGS.model_path, init_pragma_dict=None):
    num_features = dataset[0].num_features
    edge_dim = dataset[0].edge_attr.shape[1]
    if init_pragma_dict is None:
        init_pragma_dict = {'all': [1, 21]}
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=init_pragma_dict).to(FLAGS.device)
    saver.log_model_architecture(model)
    if type(model_path) is list:
        assert len(model_path) == 1, 'too many models passed to visualize'
        model_path = model_path[0]
    if model_path != None:
        saver.info(f'loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        shutil.copy(model_path, join(saver.logdir, f"{(basename(model_path)).split('.')[0]}.pth"))
    else:
        saver.error(f'model path should be set during {FLAGS.subtask}')
        raise RuntimeError()
    
    return model 

def _get_attr(emb_dict, i, target_name):
    emb = emb_dict[target_name][i]
    len_y = len(emb) if hasattr(emb, '__len__') else 1
    emb = emb.reshape(1, len_y)
    
    return emb

def encode_data_list(dataset, model, mode='tsne'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    design_points = defaultdict(lambda: defaultdict(list)) ## {kernel_name: [[emb_P, emb_T, TARGETS]]}
    design_points_all = defaultdict(list) ## {obj: [list]}

    for ind, data in enumerate(tqdm(data_loader)):
        # if ind > 3:
        #     break
        if mode == 'tsne':
            with torch.no_grad():
                data = data.to(FLAGS.device)
                out_dict, *_ = model.to(FLAGS.device)(data)
                embs = {e: out_dict[e].detach().cpu().numpy() for e in TargetEmb if e != 'whole' and e != 'pragmas'}
                embs['pragmas'] = _get_y_with_target(data, 'pragmas').detach().cpu().numpy()
                y = {target_name: _get_y_with_target(data, target_name).detach().cpu().numpy() for target_name in TARGETS}
                # y = {target_name: out_dict[target_name].detach().cpu().numpy() for target_name in TARGETS}
                    
                for ind, kernel_name in enumerate(data.gname):
                    design_points_all['kernel_name'].append(kernel_name)
                    design_points[kernel_name]['kernel_name'].append(kernel_name)
                    res = {}
                    for emb_name in TargetEmb:
                        if emb_name == 'whole': ## the other embeddings must be added before
                            cur_emb = np.copy(res[TargetEmb[0]])
                            for i in range(1, len(res)):
                                new_emb = np.copy(res[TargetEmb[i]])
                                cur_emb = np.concatenate((cur_emb, new_emb), axis=1)
                            emb_P = cur_emb
                        else:
                            emb_P = _get_attr(embs, ind, emb_name)
                        design_points_all[emb_name].append(emb_P)
                        design_points[kernel_name][emb_name].append(emb_P)
                        res[emb_name] = emb_P
                    for target_name in TARGETS:
                        y_i = _get_attr(y, ind, target_name)
                        # if 'actual_perf' in target_name:
                        #     saver.log_info(f'{ind}: {target_name}, y: {y[target_name]}, y_i {y_i}, {y[target_name][0]}, {y[target_name][1]}')
                        res[emb_name] = y_i   
                        design_points_all[target_name].append(y_i)
                        design_points[kernel_name][target_name].append(y_i)
                        
                    # design_points[kernel_name].append(res)                         
        else:
            raise NotImplementedError()

    return design_points, design_points_all

def get_embeddings(dataset, model_path=FLAGS.model_path, init_pragma_dict=None):
    model = get_model(dataset, model_path=FLAGS.model_path, init_pragma_dict=None)
    design_points, design_points_all = encode_data_list(dataset, model, mode='tsne')
    saver.info(f'{len(design_points)} distinct kernels in design points')
    
    return design_points, design_points_all


def tsne(df, feat_cols, vis_emb, dir_name=None, perplexity = 15, n_iter = 5000):
    if not FLAGS.vis_per_kernel: perplexity = 50
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    X_to_fit = df[feat_cols].values
    # df.to_pickle('test_pickle.pkl')
    print(f'Size of X_to_fit: {X_to_fit.shape}')
    tsne_results = tsne.fit_transform(X_to_fit)
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    for target_name in TARGETS + ['kernel_name']:
        if len(set(df[target_name])) <= 1:
            saver.warning(f'skipping {target_name} as it has {len(set(df[target_name]))} value')
            continue
        plt.figure(figsize=(8, 5))
        ax = plt.axes()
        ax.set_facecolor('white')
        cmap = sns.color_palette("vlag", as_cmap=True) #, n_colors=30)
        # sns.set_theme(style='dark'
        if target_name == 'kernel_name':
            cmap = sns.color_palette(cc.glasbey, n_colors=len(set(df[target_name])))
        sns_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=target_name,
            palette=cmap,
            s=30,
            data=df
            # style='kernel_name'
            # legend='full'
            # alpha=0.3
        )
        # sns_plot.set(title=f'{target_name} via {vis_emb}')
        if target_name == 'kernel_name':
            # sns_plot.legend(loc='center left',  ncol=4)
            # sns_plot.legend(loc='upper right', bbox_to_anchor=(0.0, 0.5), ncol=2)
            sns_plot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        else:
            norm = plt.Normalize(df[target_name].min(), df[target_name].max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            # Remove the legend and add a colorbar
            sns_plot.get_legend().remove()
            sns_plot.figure.colorbar(sm)
            # sns_plot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0) #, ncol=5)
            # sns_plot.subplots_adjust(right=0.7)
        plotdir = saver.plotdir if dir_name == None else join(saver.get_log_dir(), dir_name)
        create_dir_if_not_exists(plotdir)
        logdir = saver.get_log_dir() if dir_name == None else join(saver.get_log_dir(), dir_name)
        create_dir_if_not_exists(logdir)
        save_path = join(plotdir, f'tsne_{vis_emb}_{target_name}.png')
        sns_plot.get_figure().savefig(save_path, bbox_inches='tight')
        save_path_npy = join(logdir, f'tsne_{target_name}.npy')
        np.save(save_path_npy, tsne_results)
    
    return tsne_results

def cal_distance(design_points_all, dict_dist, kernel_name):
    global all_count
    for vis_emb in ['emb_P']:
        X = design_points_all[vis_emb]
        X = np.vstack(X)
        print(X.shape)
        max_dist, count, sum_dist = 0, 0, 0
        for x in X:
            for y in X[1:]:
                dist = np.linalg.norm(x-y)
                max_dist = max(max_dist, dist)
                sum_dist += dist
                count += 1
        saver.info(f'Total of {X.shape[0]} embeddings - the max distance is {max_dist:.4f} and avg dist is: {sum_dist / count:.4f}, {count}')
        dict_dist[kernel_name] = {'max': max_dist, 'avg': sum_dist / count}       
        all_count += X.shape[0] 
        saver.log_info(f'total of {all_count} embeddings so far')
        


def run_vis_emb_per_target(design_points_all, dir_name=None):
    for vis_emb in TargetEmb:
        X = design_points_all[vis_emb]
        X = np.vstack(X)
        # if X.shape[1] > 50:
        #     # first reduce dimensionality before feeding to t-sne
        #     pca = PCA(n_components=50)
        #     X = pca.fit_transform(X)
        # else:
        #     pca = PCA(n_components=9)
        #     X = pca.fit_transform(X) 
            
            
        feat_cols = ['feat' + str(i) for i in range(X.shape[1])]
        # print(feat_cols)
        df = pd.DataFrame(X, columns=feat_cols)

        for target_name in TARGETS + ['kernel_name']:
            if target_name != 'kernel_name':
                y = np.vstack(design_points_all[target_name])
                saver.log_info(f'{target_name} y.shape {y.shape}')
            else:
                y = design_points_all[target_name]
            

            df[target_name] = y

        # for i in range(X.shape[0]):
        #     saver.log_info(f"{i} X: {X[i]}, perf: {df['perf'][i]}, actual perf: {df['actual_perf'][i]}")
        saver.log_info('Size of the dataframe: {}'.format(df.shape))

        embs = tsne(df, feat_cols, vis_emb, dir_name=dir_name)
        saver.save_dict({'tsne_embs': embs, 'df': df,
                        'feat_cols': feat_cols, 'design_points': design_points_all},
                        f'{vis_emb}-tsne_results',
                        dir_name=dir_name)    


def visualize_embeddings(dataset):
    design_points_per_kernel, design_points_all = get_embeddings(dataset)
    if FLAGS.vis_per_kernel:
        dict_dist = {}
        for kernel_name in design_points_per_kernel:
            if False and kernel_name != 'gemm-blocked':
                continue
            saver.info(f'########### now processing {kernel_name} ###########')
            if 'vis' in FLAGS.subtask:
                run_vis_emb_per_target(design_points_per_kernel[kernel_name], dir_name=kernel_name)
            else:
                cal_distance(design_points_per_kernel[kernel_name], dict_dist=dict_dist, kernel_name=kernel_name)
        for name in ['max', 'avg']:
            dist = [d[name] for k, d in dict_dist.items()]
            print_stats(dist, name, saver=saver)
    else:
        saver.info(f'processing all the kernels together')
        run_vis_emb_per_target(design_points_all)
    