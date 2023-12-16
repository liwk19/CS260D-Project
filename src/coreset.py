import numpy as np
import torch
from torch_geometric.data import Dataset, DataLoader
from torch.utils.data import random_split
from os.path import join
from glob import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import copy

import config
poly_KERNEL = config.poly_KERNEL
from config import FLAGS
from model import Net
from utils import get_save_path

from torch.utils.data import ConcatDataset

TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']
SAVE_DIR = join(get_save_path(), FLAGS.dataset,  f'MLP-{FLAGS.pragma_as_MLP}-round{FLAGS.round_num}-40kernel-icmp-feb-db-{FLAGS.graph_type}-{FLAGS.task}_edge-position-{FLAGS.encode_edge_position}_norm_with-invalid_{FLAGS.invalid}-normalization_{FLAGS.norm_method}_no_pragma_{FLAGS.no_pragma}_tag_{FLAGS.tag}_{"".join(TARGET)}')

# We cannot import MyOwnDataset from programl_data due to circular import, so I copy it here
class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_files=None):
        # self.processed_dir = PROCESSED_DIR
        super(MyOwnDataset, self).__init__(SAVE_DIR, transform, pre_transform)
        if data_files is not None:
            self.data_files = data_files
        # self.SAVE_DIR = SAVE_DIR

    @property
    def raw_file_names(self):
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        # return ['data_1.pt', 'data_2.pt', ...]
        if hasattr(self, 'data_files'):
            return self.data_files
        else:
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
    
    def get_file_path(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = osp.join(SAVE_DIR, 'data_{}.pt'.format(idx))
        return fn

    def get(self, idx):
        if hasattr(self, 'data_files'):
            fn = self.data_files[idx]
        else:
            fn = osp.join(SAVE_DIR, 'data_{}.pt'.format(idx))
        data = torch.load(fn)
        return data


def get_hidden(li, pragma_dim, model=None):
    whole_dataset = MyOwnDataset(data_files=li)
    loader = DataLoader(whole_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    feat_list = []
    if FLAGS.craig_metric == 'input':
        for i in range(len(li)):
            feat_list.append(loader.dataset[i].x)
    
    else:
        if model is None:
            num_features = loader.dataset[0].num_features
            edge_dim = loader.dataset[0].edge_attr.shape[1]
            model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
            if FLAGS.model_path != None:
                model_path = FLAGS.model_path[0] if type(FLAGS.model_path) is list else FLAGS.model_path 
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        feat_list = []

        for data in tqdm(loader):
            data = data.to(FLAGS.device)
            if FLAGS.craig_metric == 'parameter':
                out_dict, loss, loss_dict_, gae_loss = model.to(FLAGS.device)(data)
            else:
                out_dict, loss, loss_dict_, gae_loss, hidden_rep = model.to(FLAGS.device)(data,
                    return_middle=int(FLAGS.craig_metric[1]))
            optimizer.zero_grad()
            loss.backward()
            if FLAGS.craig_metric == 'parameter':
                all_grad = []
                for p in model.parameters():
                    if p.grad is not None:
                        all_grad.append(p.grad.reshape(-1))
                all_grad = torch.concat(all_grad)
                feat_list.append(all_grad)
            else:
                feat_list.append(hidden_rep.reshape(-1))
        optimizer.zero_grad()
    return feat_list


def craig_split(file_li, lengths, pragma_dim, model=None, current_train=[]):
    feat_list = get_hidden(file_li, pragma_dim, model)
    feat_diff = []
    for i in range(len(file_li)):
        feat_diff.append([])
        for j in range(len(file_li)):
            feat_diff[i].append(torch.norm(feat_list[i] - feat_list[j]).item())
    feat_diff = torch.tensor(feat_diff)

    train_len, val_len, test_len = lengths[0], lengths[1], lengths[2]
    selected_indices = current_train
    for round in range(train_len - len(current_train)):
        best_cand = -1
        min_diff = 1e10
        for new_ind in range(len(file_li)):
            if new_ind in selected_indices:
                continue
            new_diff = feat_diff[[new_ind] + selected_indices]
            new_diff = torch.sum(new_diff.min(0).values)
            if new_diff < min_diff:
                min_diff = new_diff
                best_cand = new_ind
        assert best_cand >= 0
        selected_indices.append(best_cand)
    print(selected_indices)
    # np.save(f'hidden_save/{poly_KERNEL[0]}_{FLAGS.craig_metric}.npy', selected_indices)
    # exit()
    
    best_diff = feat_diff[selected_indices].min(0).values
    weight = []
    for i in selected_indices:
        weight.append((feat_diff[i] == best_diff).sum() / len(best_diff))
    weight = torch.tensor(weight, device=FLAGS.device)
    
    train_li = [file_li[i] for i in selected_indices]
    unselected_li = []
    for i, li in enumerate(file_li):
        if i not in selected_indices:
            unselected_li.append(li)
    li = random_split(unselected_li, [val_len, test_len], generator=torch.Generator().manual_seed(FLAGS.random_seed))
    return [train_li, li[0], li[1]], weight, selected_indices

def uncertainty_split(file_li, lengths, uncertain_indices=[]):
    train_len, test_len = lengths[0], lengths[1]
    train_li = [file_li[i] for i in uncertain_indices]
    unselected_li = []
    for i, li in enumerate(file_li):
        if i not in uncertain_indices:
            unselected_li.append(li)
    # li = random_split(unselected_li, [val_len, test_len], generator=torch.Generator().manual_seed(FLAGS.random_seed))
    return [train_li, unselected_li]
        

def tsne_plot(li, epoch, pragma_dim, model=None):
    train_len = len(li[0])
    print(train_len)
    all_li = copy.deepcopy(li[0])
    all_li.extend(li[1])
    all_li.extend(li[2])
    print(len(all_li))
    # train_visualize_loader = DataLoader(li[0], batch_size=1, pin_memory=True, num_workers=4)
    # val_visualize_loader = DataLoader(li[1], batch_size=1, pin_memory=True, num_workers=4)
    # test_visualize_loader = DataLoader(li[2], batch_size=1, pin_memory=True, num_workers=4)
    # train_input = None
    # test_input = None
    # val_input = None
    # for data in train_visualize_loader:
    #     train_input = data.x
    # for data in val_visualize_loader:
    #     val_input = data.x
    # for data in test_visualize_loader:
    #     test_input = data.x
    feat_list = get_hidden(all_li, pragma_dim, model)
    vis_input = torch.stack(feat_list).cpu().numpy()
    vis_emb = TSNE(n_components=2, perplexity=30, learning_rate='auto').fit_transform(vis_input)
    plt.scatter(vis_emb[train_len:,0], vis_emb[train_len:,1], c='blue')
    plt.scatter(vis_emb[0:train_len,0], vis_emb[0:train_len,1], c='red')
    plt.title(f'{poly_KERNEL[0]} {FLAGS.craig_metric} Epoch={epoch}')
    plt.savefig(f'{poly_KERNEL[0]}_{FLAGS.craig_metric}_{epoch}.png', dpi=1200)
    plt.close()


def find_uncertain_indices(li, num_features, edge_dim, pragma_dim):
    '''
    returns selected 20 data points
    '''
    all_model_out = []

    uncertainty_loader = DataLoader(li, batch_size=1, pin_memory=True, num_workers=4)  # TODO
    for model_idx in range(len(FLAGS.ensemble_KERNEL)):
        out_list = []
        model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)
        model_path = f'{FLAGS.ensemble_model_path}_{FLAGS.ensemble_KERNEL[model_idx]}/run1/'
        model_path += f'{FLAGS.ensemble_model_epoch}_train_model_state_dict.pth'
        print(f'=== loaded {model_idx + 1} model ====')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            for data in uncertainty_loader:
                data = data.to(FLAGS.device)
                out_dict, _, _, _ = model.to(FLAGS.device)(data)
                out = [out_dict['perf'].item(), out_dict['util-LUT'].item(), out_dict['util-FF'].item(), 
                        out_dict['util-DSP'].item(), out_dict['util-BRAM'].item()]
                out_list.append(out)
        all_model_out.append(out_list)

    all_model_out = torch.tensor(all_model_out)
    model_out_std = all_model_out.std(0) 
    model_out_std = model_out_std.mean(1)
    if FLAGS.uncertainty_metric == 'high_uncertain':
        _, indices = torch.topk(model_out_std, FLAGS.transfer_k_shot)
    elif FLAGS.uncertainty_metric == 'low_uncertain':
        _, indices = torch.topk(-model_out_std, FLAGS.transfer_k_shot)
    elif FLAGS.uncertainty_metric == 'hybrid':
        assert FLAGS.uncertainty_split[0] + FLAGS.uncertainty_split[1] == FLAGS.transfer_k_shot
        _, low_uncertainty_indices = torch.topk(model_out_std, FLAGS.uncertainty_split[0], largest=False)
        _, high_uncertainty_indices = torch.topk(model_out_std, FLAGS.uncertainty_split[1])
        if not FLAGS.uncertainty_split_alternating:    
            indices = torch.cat((low_uncertainty_indices, high_uncertainty_indices))
        else:
            # generate low, high, low, high points
            batch = 5
            batch_size = int(FLAGS.transfer_k_shot/batch)
            low_batch = int(FLAGS.uncertainty_split[0]/batch)
            high_batch = int(FLAGS.uncertainty_split[1]/batch)
            indices = torch.tensor([], dtype=torch.int32)
            for i in range(int(batch)):
                indices = torch.cat((indices, low_uncertainty_indices[i*batch_size:i*batch_size+low_batch+1]))
                indices = torch.cat((indices, high_uncertainty_indices[i*batch_size:i*batch_size+high_batch+1]))
        print(indices)
    return indices
