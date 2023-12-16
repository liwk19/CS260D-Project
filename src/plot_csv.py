from audioop import mul
import csv
from saver import saver
from utils import plot_scatter_with_subplot, plot_scatter_with_subplot_trend

## parallel w unsafe
file_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/src/logs/auto-encoder/all-data-sepPT/correct-edge-ID/simple-program/unsafe-math-134f-SSL-False-gae-T-True-gae-P-False-test_2022-08-04T00-09-16.797829/actual-prediction.csv'
## parallel wo unsafe
file_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/src/logs/auto-encoder/all-data-sepPT/correct-edge-ID/simple-program/wo-unsafe-134f-SSL-False-gae-T-True-gae-P-False-test_2022-08-04T00-14-06.595020/actual-prediction.csv'
## reduction wo unsafe
file_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/src/logs/auto-encoder/all-data-sepPT/correct-edge-ID/simple-program/reduction-wo-unsafe-134f-SSL-False-gae-T-True-gae-P-False-test_2022-08-04T10-57-48.064086/actual-prediction.csv'
## gemm-blocked from machsuite
file_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/src/logs/auto-encoder/all-data-sepPT/correct-edge-ID/simple-program/blocked-reduction-wo-unsafe-134f-SSL-False-gae-T-True-gae-P-False-test_2022-08-04T11-37-15.467372/actual-prediction.csv'

pragma = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 1024.0, 2048.0]         
size = [256, 512, 1024, 2048]
precision = ['float', 'double']   
multi_target = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']

def get_csv_dict():
    csv_dict = {'header': []}
    with open(file_path, 'r') as fp:
        csv_reader = csv.DictReader(fp)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                csv_dict['header'] = row.keys()
            
            gname = row['gname']
            pragma = row['pragma']
            csv_dict[f'{gname}-{pragma}'] = row
    return csv_dict


csv_dict = get_csv_dict()   

single_plot = True
points_dict_multi_target = {}
trained = True
connected = True
if trained:
    gname = 'gemm-blocked'
    title = gname
    connected = True
    num = 0
    if not single_plot:
        points_dict_multi_target = {}
    for target in multi_target:
        if target not in points_dict_multi_target:
            points_dict_multi_target[target] = {'pred': {}, 'true': {}}
        # print(points_dict_multi_target[target]['pred'].keys())
        for pp in pragma:
            for key, val in csv_dict.items():
                if f'{gname}-{pp}' in key:
                    title = '-'.join(val['pragma'].split('-')[1:])
                    if title not in points_dict_multi_target[target]['pred']:
                        num += 1
                        points_dict_multi_target[target]['pred'][f'{title}'] = []
                        points_dict_multi_target[target]['true'][f'{title}'] = []
                    points_dict_multi_target[target]['pred'][f'{title}'].append((pp, csv_dict[key][f'predicted-{target}']))
                    points_dict_multi_target[target]['true'][f'{title}'].append((pp, csv_dict[key][f'acutal-{target}']))
                    
    print(num / len(multi_target))
else:
    for p in precision:
        for s in size:
            gname = f'dot-{p}-{s}'
            title = f'{p[0]}{s}'
            if not single_plot:
                points_dict_multi_target = {}
            for target in multi_target:
                if target not in points_dict_multi_target:
                    points_dict_multi_target[target] = {'pred': {}, 'true': {}}
                if title not in points_dict_multi_target[target]['pred']:
                    points_dict_multi_target[target]['pred'][f'{title}'] = []
                    points_dict_multi_target[target]['true'][f'{title}'] = []
                # print(points_dict_multi_target[target]['pred'].keys())
                for pp in pragma:
                    if f'{gname}-{pp}' in csv_dict:
                        points_dict_multi_target[target]['pred'][f'{title}'].append((pp, csv_dict[f'{gname}-{pp}'][f'predicted-{target}']))
                        points_dict_multi_target[target]['true'][f'{title}'].append((pp, csv_dict[f'{gname}-{pp}'][f'acutal-{target}']))
            if not single_plot:
                plot_scatter_with_subplot(points_dict_multi_target, f'{gname}', saver.plotdir, multi_target) 
            
if single_plot:
    plot_scatter_with_subplot_trend(points_dict_multi_target, f'all', saver.plotdir, multi_target, connected = connected) 

        
# print(csv_dict['header'])
# print(csv_dict['dot-float-1024-4.0'])
