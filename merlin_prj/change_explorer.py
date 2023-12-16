import json
import argparse
import shutil
from os.path import join, exists

MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil_stencil2d',
                    'nw', 'md', 'stencil-3d']

poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen', 
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver', 
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm', 
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium']
ALL_KERNELS = {'machsuite': MACHSUITE_KERNEL, 'poly': poly_KERNEL}

def change_exp(f, exp, Tdse, Thls, Tmerlin):
    jf = json.load(open(f, 'r'))
    jf['search.algorithm.name']=str(exp)
    jf['timeout.exploration']=int(Tdse)
    jf['timeout.hls']=int(Thls)
    jf['timeout.transform']=int(Tmerlin)
    with open(f, "w") as outfile:
        json.dump(jf, outfile, indent=4)
        
def copy_database(src, dst, kernel):
    shutil.copyfile(src, join(dst, f'{kernel}_result_1.db'))
    
        
def arg_parser() -> argparse.Namespace:
    """Parse user arguments."""

    parser_run = argparse.ArgumentParser(description='Changing design space config')
    parser_run.add_argument('--file-path',
                        required=True,
                        action='store',
                        help='path to the design space config')
    parser_run.add_argument('--explorer',
                        required=True,
                        action='store',
                        help='type of the explorer')
    parser_run.add_argument('--dse-time',
                        required=False,
                        action='store',
                        default='1440',
                        help='DSE timeout')
    parser_run.add_argument('--hls-time',
                        required=False,
                        action='store',
                        default='100',
                        help='HLS timeout')
    parser_run.add_argument('--merlin-time',
                        required=False,
                        action='store',
                        default='20',
                        help='Merlin timeout')
    parser_run.add_argument('--mode',
                        required=False,
                        action='store',
                        default='dse',
                        help='Modify the explorer or copy database files')
    

    return parser_run.parse_args()
        
if __name__ == '__main__':
    args = arg_parser()
    if args.mode == 'dse':
        change_exp(args.file_path, args.explorer, args.dse_time, args.hls_time, args.merlin_time)
    elif args.mode == 'copy':
        src = '/home/atefeh/database/new_merlin-2021.1/brady'
        for benchmark in ['machsuite', 'poly']:
            if benchmark == 'poly':
                dst = f'/home/atefeh/software-gnn/dse_database/poly/databases/v21/archive/{args.explorer}'
            else:
                dst = f'/home/atefeh/software-gnn/dse_database/machsuite/databases/v21/original-size/archive/{args.explorer}'
                
            for kernel in ALL_KERNELS[benchmark]:
                if exists(join(src, kernel)):
                    copy_database(join(src, kernel, 'result.db'), dst, kernel)
    else:
        raise NotImplementedError()
    
    