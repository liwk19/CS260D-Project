import matplotlib
# matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from os.path import join
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle

objectives = ['perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM']

num_models = 5

def get_objectives(df, keyword):
    res = {}
    for o in objectives:
        res[o] = df[f'{keyword}-{o}'].to_numpy()
    return res

def get_predictions():
    base_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/src/logs/auto-encoder/all-data-sepPT/round1/task-transfer/ensemble-meta-weight/predictions/norm-perf-edge-attr-True-position-True-6L-SSL-False-gae-T-True-gae-P-False-test_regression_inference_2023-01-04T23-22-42.900521'
    dataset_dict = {}
    for i in range(num_models):
        # Load CSV and columns
        df = pd.read_csv(join(base_path, f'actual-prediction-{i}.csv'))
        if i == 0:
            dataset_dict['actual'] = get_objectives(df, 'acutal')
        dataset_dict[f'prediction-{i}'] = get_objectives(df, 'predicted')
    return dataset_dict


def convert_input_numpy(data):
    return data.reshape((len(data), 1))

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset():
    dataset_dict = get_predictions()
    stackX = {o: None for o in objectives}
    Y = {o: None for o in objectives}
    for i in range(num_models):
        # make prediction
        for o in objectives:
            yhat = convert_input_numpy(dataset_dict[f'prediction-{i}'][o])
            # print(yhat.shape)
            # stack predictions into [rows, members, probabilities]
            if stackX[o] is None:
                stackX[o] = yhat
            else:
                stackX[o] = np.hstack((stackX[o], yhat))
            # print(stackX[o].shape)
    for o in objectives:
        Y[o] = convert_input_numpy(dataset_dict[f'actual'][o])
    for o in objectives:
        print(stackX[o].shape)
        print(Y[o].shape)
    # flatten predictions to [rows, members x probabilities]
    # stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX, Y

def report_accuracy(y, y_pred, csv_dict, label, objective):
    # The mean squared error
    print(y.shape, y_pred.shape)
    print("Mean squared error: %.4f" % mean_squared_error(y, y_pred, squared=True))
    # The root mean squared error
    print("Root mean squared error: %.4f" % mean_squared_error(y, y_pred, squared=False))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.4f" % r2_score(y, y_pred))
    # label = label + '-' + objective
    # csv_dict[label] = {}
    # csv_dict[label]['model'] = label
    # csv_dict[label]['MSE'] = "%.4f" % mean_squared_error(y, y_pred, squared=True)
    # csv_dict[label]['RMSE'] = "%.4f" % mean_squared_error(y, y_pred, squared=False)
    # csv_dict[label]['R2'] = "%.4f" % r2_score(y, y_pred)
    
    csv_dict[label]['model'] = label
    csv_dict[label][objective] = "%.4f" % mean_squared_error(y, y_pred, squared=False)
    # csv_dict[label]['R2'] = "%.4f" % r2_score(y, y_pred)
     


def test_model(model, x, y, csv_dict, objective):
    # Make predictions using the testing set
    y_pred = model.predict(x)

    # The coefficients
    print("Coefficients: \n", model.coef_)
    report_accuracy(y, y_pred, csv_dict, 'meta-learner', objective)
    
def log_dict_of_dicts_to_csv(fn, csv_dict, csv_header, delimiter=','):
    import csv
    fp = open(join('', f'{fn}.csv'), 'w+')
    f_writer = csv.DictWriter(fp, fieldnames=csv_header)
    f_writer.writeheader()
    for d, value in csv_dict.items():
        if d == 'header':
            continue
        f_writer.writerow(value)
    fp.close()

# fit a model based on the outputs from the ensemble members
def fit_stacked_model():
    # create dataset using ensemble
    stackX, Y = stacked_dataset()
    model_dict = {}
    csv_dict = {'header' : ['model', 'perf', 'util-LUT', 'util-FF', 'util-DSP', 'util-BRAM', 'total']}
    csv_dict['meta-learner'] = {}
    for i in range(num_models):
        csv_dict[f'base-learner-{i}'] = {}
    for o in ['perf']:
    # for o in objectives:
        # fit standalone model
        model = linear_model.LinearRegression()
        # Y[o], stackX[o] = shuffle(Y[o], stackX[o])
        split_train = 250
        model.fit(stackX[o][:split_train], Y[o][:split_train])
    for o in objectives:
        print('######################')
        print(f'Testing for {o}')
        split_train = 0
        test_model(model, stackX[o][split_train:], Y[o][split_train:], csv_dict, o)
        for i in range(num_models):
            print('base learner', i)
            X = stackX[o][split_train:]
            X = X.transpose()
            report_accuracy(Y[o][split_train:], X[i], csv_dict, f'base-learner-{i}', o)
        model_dict[o] = model
    for i in range(num_models):
        sum = 0
        for o in objectives:
            sum += float(csv_dict[f'base-learner-{i}'][o])
        csv_dict[f'base-learner-{i}']['total'] = sum
    sum = 0
    for o in objectives:
        sum += float(csv_dict[f'meta-learner'][o])
    csv_dict[f'meta-learner']['total'] = sum
    log_dict_of_dicts_to_csv(f'ensemble-weighted-meta-rmse-perf', csv_dict, csv_dict['header'])
    return model_dict


fit_stacked_model()


