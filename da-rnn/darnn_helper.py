import numpy as np
import pandas as pd
import numpy as np
import signalz
from darnn_model import DA_rnn
from sklearn import preprocessing

def get_google_rev():
    total_df = pd.read_csv('C:\\Users\\Hawk\\DS\\project\\total_df.csv')
    cols_to_remove = ['totals.sessionQualityDim', 'totals.timeOnSite', 'totals.totalTransactionRevenue', 'totals.transactions', 'fullVisitorId', 'visitId', 'totals.transactionRevenue']
    y = total_df['totals.transactionRevenue'].values

    total_df.drop(cols_to_remove, axis=1, inplace=True)
    return total_df.values, y

def get_ibm_stocks(project, column='Close'):
    dataset = pd.read_csv('data\\IBM_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
    signal = dataset.loc[:, column].values[:2600]

    signal = np.array(signal)
    signal = scale(signal)
    X = signal[:-1]
    y = signal[1:]
    X = X[:, None]
    return train_val_test_split(X, y, project)

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
def get_henon_series(project, elements=16000):
    data_str = open("data\\HENON.DAT", "r").read()
    data = list(map(float, data_str.splitlines()))[:elements]
    data = np.array(data)
    X = data[:-1]
    y = data[1:]
    X = X[:, None]
    return train_val_test_split(X, y, project)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def get_mackey_glass(length=1000, add_noise=False, noise_range=(-0.01, 0.01)):
    r = np.random.RandomState(42)
    initial = .25 + .5 * r.rand()
    signal = signalz.mackey_glass(length, a=0.2, b=0.8, c=0.9, d=23, e=10, initial=initial)
    if add_noise:
        signal += r.uniform(noise_range[0], noise_range[1], size=signal.shape)

    x = signal[:-1]
    y = signal[1:]
    x =  x[:,None]
    return x,y

def get_mackey_glass_series(elements=10000):
    data_str = open("data\\MackeyGlass_t17.txt", "r").read()
    data = list(map(float, data_str.splitlines()))[:elements]
    data = np.array(data)
    X = data[:-1]
    y = data[1:]
    X = X[:, None]
    return X, y
    
def get_mackey_glass_series_temp(project, elements=10000):
    data_str = open("data\\MackeyGlass_t17.txt", "r").read()
    data = list(map(float, data_str.splitlines()))[:elements]
    data = np.array(data)
    X = data[:-1]
    y = data[1:]
    X = X[:, None]
    return train_val_test_split(X, y, project)

def construct_model(X, y, ntimestep=10, batchsize=128, eta=0.1, nEncoder=64, epochs = 100):
    nhidden_encoder = nhidden_decoder = nEncoder
    #train_split = 0.8096667845223008
    train_split = 0.7
    input_size = X.shape[1] # Number of features
    # Initialize model
    return DA_rnn(input_size, X, y, ntimestep, nhidden_encoder, nhidden_decoder, batchsize, eta, epochs, train_split)

    #don't need it now
def train_val_test_split(X, y, project):


    train_split = X.shape[0] - project
    # Train set
    X_train = X[0:train_split, :]
    y_train = y[0:train_split]

    # Test set
    X_test = X[train_split:, :]
    y_test = y[train_split:]

    return X_train, y_train, X_test, y_test
