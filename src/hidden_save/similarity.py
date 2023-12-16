import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


ensemble_KERNEL = ['gemm-blocked', 'gemm-ncubed', 'spmv-ellpack', 'stencil', 'stencil-3d', 'nw',
        'doitgen', 'doitgen-red', '2mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance',
        'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gesummv', 'heat-3d', 'jacobi-2d',
        'seidel-2d', 'symm', 'symm-opt', 'syrk', 'syr2k', 'trmm', 'mvt-medium', 'correlation',
        'atax-medium', 'bicg-medium', 'symm-opt-medium', 'gesummv-medium', 'gemver-medium']
# test_KERNEL = ['jacobi-1d', 'fdtd-2d', 'trmm-opt', '3mm', 'gemver', 'mvt']
test_KERNEL = ['mvt']


def visualize():
    hiddens = []
    lengths = []
    for k in ensemble_KERNEL:
        new_hidden = np.load(f'{k}.npy')[:50]
        hiddens.append(new_hidden)
        lengths.append(len(new_hidden))
    hiddens = np.concatenate(hiddens, 0)
    test_hiddens = []
    test_lengths = []
    for t in test_KERNEL:
        new_hidden = np.load(f'{t}.npy')[:50]
        test_hiddens.append(new_hidden)
        test_lengths.append(len(new_hidden))
    test_hiddens = np.concatenate(test_hiddens, 0)
    hiddens = np.concatenate([hiddens, test_hiddens], 0)
    vis_emb = TSNE(n_components=2, perplexity=30, learning_rate='auto').fit_transform(hiddens)

    len_sum = 0
    plt.figure(figsize=(10, 10))
    for i, new_len in enumerate(lengths):
        plt.scatter(vis_emb[len_sum:len_sum+new_len,0], vis_emb[len_sum:len_sum+new_len,1], color='black')
        len_sum += new_len
    for i, new_len in enumerate(test_lengths):
        plt.scatter(vis_emb[len_sum:len_sum+new_len,0], vis_emb[len_sum:len_sum+new_len,1], label=test_KERNEL[i])
        len_sum += new_len
    plt.legend()
    plt.savefig(f'visual.png', dpi=1200, bbox_inches='tight')
    plt.close()


def cos_similar(v1, v2):
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom != 0 else 0


def calculate_sim():
    ensemble_hiddens = []
    for k in ensemble_KERNEL:
        try:
            ensemble_hiddens.append(np.load(f'{k}.npy').mean(0))
        except:
            pass
    for t in test_KERNEL:
        test_hidden = np.load(f'{t}.npy').mean(0)
        print(test_hidden.shape)
        dis_list = []
        for h in ensemble_hiddens:
            # dis = np.linalg.norm(test_hidden - h)
            dis = cos_similar(h, test_hidden)
            dis_list.append(dis)
        print(f'{t}: {np.mean(dis_list):.3f}, {np.min(dis_list):.3f}, {np.max(dis_list):.3f}')


def visualize_coreset():
    for t in test_KERNEL:
        hiddens = np.load(f'{t}.npy')
        vis_emb = TSNE(n_components=2, perplexity=30, learning_rate='auto').fit_transform(hiddens)
        # for metric in ['h5', 'h6', 'h7', 'h8']:
        #     plt.scatter(vis_emb[:, 0], vis_emb[:, 1])
        #     coreset = np.load(f'{t}_{metric}.npy')
        #     plt.scatter(vis_emb[coreset, 0], vis_emb[coreset, 1], label=metric)
        #     plt.legend()
        #     plt.savefig(f'visual_{t}_{metric}.png', bbox_inches='tight')
        #     plt.close()
        for metric in ['high_uncertain', 'low_uncertain']:
            plt.scatter(vis_emb[:, 0], vis_emb[:, 1])
            coreset = np.load(f'../ensemble_data_split/{t}_{metric}.npy')
            coreset = [int(c.split('_')[-1].split('.')[0]) for c in coreset][:50]
            plt.scatter(vis_emb[coreset, 0], vis_emb[coreset, 1], label=metric)
            plt.legend()
            plt.savefig(f'visual_{t}_{metric}.png', bbox_inches='tight')
            plt.close()


def main():
    # calculate_sim()
    # visualize()
    visualize_coreset()


if __name__ == '__main__':
    main()
