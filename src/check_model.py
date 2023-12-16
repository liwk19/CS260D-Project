import torch
from utils import get_root_path, load
from glob import iglob
from os.path import join
from model import Net
from config import FLAGS

from os.path import join, dirname

models_path = [f for f in iglob(join(get_root_path(), 'save/programl/**'), recursive = True) if f.endswith('.pth') and 'node-att' in f]
models_path = [f for f in iglob(join(get_root_path(), 'src/logs/auto-encoder/**'), recursive = True) if f.endswith('.pth') and 'val-test' in f]
models_path = ['/home/atefehSZ/github-software-gnn/software-gnn/src/logs/auto-encoder/all-data-sepPT/round12-34kernel/debug-pytorch-loss-larger-trainset-split-kernel-test-different-seed-correct-graph-type_norm-perf-edge-attr-True_position-True-6L-SSL-False-gae-T-True-gae-P-False-test_regression_train_2023-02-22T11-48-26.494450/run1/val_model_state_dict.pth']
models_path = ['/share/atefehSZ/RL/original-software-gnn/software-gnn/models/iccad/v18/2l-parallel-val_model_state_dict-0.pth']
models_path = ['/home/atefeh/software-gnn/models/iccad/v18/best_post-gnn-2l-parallel-val_model_state_dict-0.pth', 
               '/home/atefeh/software-gnn/models/iccad/v20/best_post-gnn-2l-parallel-freeze1_val_model_state_dict.pth']

for model in models_path:
    loaded_model = torch.load(model, map_location=torch.device('cpu'))
    
    # loaded_model_state_dict = loaded_model.load_state_dict(loaded_model)
    print('##############')
    print(model)
    print('##############')
    # print('{}\n'.format(loaded_model))
    # print('##############')
    for param in loaded_model:
        if 'weight' in param:
            print(param, loaded_model[param].size())
        
    print(loaded_model.__dict__)
        
    for param in loaded_model:
        if 'MLP' in param and '.3.' in param:
            print(param)
            print(loaded_model[param])

    print()
    print('##########################################################')
    print()
    # break

check_model_size = True
if check_model_size:    
    for model_path in models_path:   
        pragma_dim = 0
        if FLAGS.encoder_path is not None:
            pragma_dim = load(join(dirname(FLAGS.encoder_path), 'v18_pragma_dim'))
        model = Net(FLAGS.num_features, edge_dim=FLAGS.edge_dim, init_pragma_dict=pragma_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model_total_params = sum(p.numel() for p in model.parameters())
        model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{model_total_params=}')
        print(f'{model_trainable_params=}')