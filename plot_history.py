from matplotlib.pyplot import clf
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(net_history):
    history = net_history.history
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['accuracy']
    val_accuracies = history['val_accuracy']

    clf()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])

    plt.figure()
    clf()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['accuracy', 'val_accuracy'])


def loss_frame(loss_test, his):
    loss_train = his.history['loss'][-1]
    loss_val = his.history['val_loss'][-1]
    df = pd.DataFrame([loss_train, loss_val, loss_test], index=['train', 'val', 'test'], columns=['loss'])
    print(df)
