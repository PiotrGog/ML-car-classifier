import matplotlib.pyplot as plt
import pickle
import os


def save_history(history, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(history.history, f)


def load_history(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_history_plots(history, file_name=None):
    print(history.keys())
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if not file_name is None:
        filename, file_extension = os.path.splitext(file_name)
        plt.savefig(filename + "_accuracy" + file_extension)
    plt.close()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if not file_name is None:
        filename, file_extension = os.path.splitext(file_name)
        plt.savefig(filename + "_loss" + file_extension)
    plt.close()


def draw_history_plots(history, file_name=None):
    print(history.keys())
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if not file_name is None:
        filename, file_extension = os.path.splitext(file_name)
        plt.savefig(filename + "_accuracy" + file_extension)
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if not file_name is None:
        filename, file_extension = os.path.splitext(file_name)
        plt.savefig(filename + "_loss" + file_extension)
    plt.show()
