import matplotlib.pyplot as plt
import os
import numpy as np


def plot(kernel):
    filenames = os.listdir('test_losses')
    res_10, res_20, res_50, res_100 = [], [], [], []

    for name in filenames:
        name_l = name.split('_')
        if name_l[0] == kernel:
            if name_l[1] == '10':
                res_10.append(np.load(f'test_losses/{name}'))
            elif name_l[1] == '20':
                res_20.append(np.load(f'test_losses/{name}'))
            elif name_l[1] == '50':
                res_50.append(np.load(f'test_losses/{name}'))
            elif name_l[1] == '100':
                res_100.append(np.load(f'test_losses/{name}'))
    res_10 = np.mean(res_10, 0)
    res_20 = np.mean(res_20, 0)
    res_50 = np.mean(res_50, 0)
    res_100 = np.mean(res_100, 0)

    plt.plot(res_10, label='k=10')
    plt.plot(res_20, label='k=20')
    plt.plot(res_50, label='k=50')
    plt.plot(res_100, label='k=100')
    plt.ylabel('Loss', size=16)
    plt.xlabel('Epoch', size=16)
    plt.legend(prop={'size': 14})
    plt.savefig(f'test_losses/{kernel}.png', bbox_inches='tight')


def output(kernel, coreset):
    val_files = os.listdir('val_losses')
    random_list = []
    hi_list = []
    for i in range(8):
        hi_list.append([])
    parameter_list = []
    input_list = []

    for name in val_files:
        name_l = name.split('_')
        if name_l[0] == kernel and name_l[1] == coreset:
            val_loss = np.load(f'val_losses/{name}')
            test_loss = np.load(f'test_losses/{name}')
            test_res = test_loss[np.argmin(val_loss)]
            test_res2 = test_loss[74]
            test_res = [test_res, test_res2]
            if name_l[2] == 'parameter':
                parameter_list.append(test_res)
            elif name_l[2] == 'input':
                input_list.append(test_res)
            elif name_l[2] == 'random':
                random_list.append(test_res)
            else:
                assert name_l[2][0] == 'h'
                hi_list[int(name_l[2][1]) - 1].append(test_res)
    
    # assert len(parameter_list) >= 2
    # for i in range(8):
    #     assert len(hi_list[i]) >= 2

    test_files = os.listdir('test_losses')
    res_20, res_50, res_100 = [], [], []
    for name in test_files:
        name_l = name.split('_')
        if name_l[0] == kernel:
            if name_l[1] == '20':
                res_20.append(np.load(f'test_losses/{name}')[74])
            elif name_l[1] == '50':
                res_50.append(np.load(f'test_losses/{name}')[74])
            elif name_l[1] == '100':
                res_100.append(np.load(f'test_losses/{name}')[74])
    print(f'random (100): {np.mean(res_100)}+-{np.std(res_100)}')
    print(f'random (50): {np.mean(res_50)}+-{np.std(res_50)}')
    print(f'random (20): {np.mean(res_20)}+-{np.std(res_20)}')
    print(f'random: {np.mean(random_list, 0)}+-{np.std(list(random_list), 0)}')
    print(f'param: {np.mean(parameter_list, 0)}+-{np.std(list(parameter_list), 0)}')
    print(f'input: {np.mean(input_list, 0)}+-{np.std(list(input_list), 0)}')
    for i in range(8):
        print(f'h{i+1}: {np.mean(hi_list[i], 0)} +- {np.std(hi_list[i], 0)}')


if __name__ == '__main__':
    kernel = 'fdtd-2d'
    coreset = 'CREST'
    # plot(kernel)
    output(kernel, coreset)
