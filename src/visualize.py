import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from os.path import join


target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']
target_o = ['Latency', 'LUT', 'FF', 'DSP', 'BRAM']
target = ['v18-perf', 'v20-perf', 'v18-util-LUT', 'v20-util-LUT', 
        'v18-util-FF', 'v20-util-FF', 'v18-util-DSP', 'v20-util-DSP', 
        'v18-util-BRAM', 'v20-util-BRAM']
# target = ['v18-perf', 'v20-perf', 'v18-LUT', 'v20-LUT', 
#         'v18-FF', 'v20-FF', 'v18-DSP', 'v20-DSP', 
#         'v18-BRAM', 'v20-BRAM']

store_data_csv = False
if store_data_csv:    
    from utils import _get_y_with_target
    from data import MyOwnDataset
    dataset = MyOwnDataset()
    print(len(dataset))
    all_data = []
    for data in dataset:
        data_list = []
        for t in target:
            data_y = _get_y_with_target(data, t).item()
            # print(data_y)
            data_list.append(data_y)
        all_data.append(data_list)
    
    with open('data_info.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(target)
        writer.writerows(all_data)
        
data_df = pd.read_csv('../dse_database/common_v18_v20.csv')
# data_df = pd.read_csv('common_v18_v20.csv')
print(data_df.head())
objectives = list(data_df.columns[:].values.tolist())

plot_type = 'corrn'
if plot_type == 'corr':
    correlations = data_df[target].corr()
    fig = plt.figure()
    fig.set_figheight(12.6)
    fig.set_figwidth(12.6)
    fig.tight_layout()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=0, vmax=1, cmap='RdBu')
    for (i, j), z in np.ndenumerate(correlations):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    fig.colorbar(cax)
    ticks = np.arange(0,len(objectives),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(target)
    ax.set_yticklabels(target)
    ax.figure.savefig('correlation_matrix-v18.png', bbox_inches='tight')
    print('correlation matrix is:')
    print(correlations)
    # plt.show()
else:
    scatter_matrix(data_df[target[:2]], alpha=0.3, figsize=(16,16), diagonal='kde', marker='o')
    plt.savefig('scatter_matrix-v18.png') #), bbox_inches='tight')
    # plt.show()