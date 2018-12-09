import matplotlib.pyplot as plt
import time

import darnn_helper
from sklearn.metrics import mean_squared_error

def plot_results(model, y_pred, series):
    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig(series+ "_1.png")
    plt.close(fig1)
    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig(series+"_2.png")
    plt.close(fig2)
    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.y[model.train_timesteps:], label="True")
    plt.legend(loc='upper left')
    plt.savefig(series+"_3.png")
    plt.close(fig3)
    
def get_error(true, pred):
    return mean_squared_error(true, pred)

def get_evaluations(true, pred):
    thirty = sixty = ninety = 0
    for i in range(1, 101):
        thirty += get_error(true[:30*i], pred[:30*i])
        sixty += get_error(true[:60*i], pred[:60*i])
        ninety += get_error(true[:90*i], pred[:90*i])

    print(thirty/100, sixty/100, ninety/100)

if __name__ == "__main__":
    # Read dataset
    X, y = darnn_helper.get_google_rev()

    # Read dataset
    tme_step = 10
    batchsize = 512
    eta = 0.01
    nEncoder = 64
    epochs = 10

    start = time.time()

    # Read dataset
    model = darnn_helper.construct_model(X, y, tme_step, batchsize, eta, nEncoder, epochs)
    # Train
    model.train()
    # Prediction
    y_pred = model.test()

    end = time.time()
    total_time = (end - start)

    error = get_error(model.y[model.train_timesteps:], y_pred)
    plot_results(model, y_pred, 'google_Rev')

    print('Finished Evaluation of DA_RNN on google_Rev, MSE = {} and total time = {}seconds'.format(error, total_time))
