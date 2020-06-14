import matplotlib.pyplot as plt
import pickle
import os


def save_history(history, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(history, f)


def load_history(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_history_plots(history, file_name=None):
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


if __name__ == "__main__":
    hist_dir = "./hist"
    tmp_dir = "./tmp"
    hist_files = [
        (os.path.join(hist_dir, f), f) for f in os.listdir(hist_dir)
    ]
    hist_files.sort(key=lambda x: x[1])
    print(hist_files)
    # exit()

    def image_insert(filename):
        splitted = filename.split("_")
        # print(splitted)
        text = f"\\begin{{figure}}[H]\n" + \
            "   \centering\n" + \
            "   \\begin{subfigure}[b]{0.48\\textwidth}\n" + \
            "       \centering\n" + \
            f"       \includegraphics[width=\\textwidth]{{wykresy/{filename}_loss.png}}\n" + \
            "       \caption{Wartość funkcji straty w przebiegu uczenia}\n" + \
            f"       \label{{subfig:{filename}_loss}}\n" + \
            "   \end{subfigure}\n" + \
            "   \\begin{subfigure}[b]{0.48\\textwidth}\n" + \
            "       \centering\n" + \
            f"       \includegraphics[width=\\textwidth]{{wykresy/{filename}_accuracy.png}}\n" + \
            "       \caption{Stopień dokładności modelu w przebiegu uczenia}\n" + \
            f"       \label{{subfig:{filename}_accuracy}}\n" + \
            "   \end{subfigure}\n" + \
            f"   \caption{{Model: {splitted[0]}; wielkość obrazka: {splitted[3]}; rozmiar mini-batcha: {splitted[4]}; współczynnik uczenia: {splitted[6]}}}\n" + \
            f"    \label{{fig:{filename}}}\n" + \
            "\end{figure}"

        print(text)
        print()
        print()

    for (hf_path, hf) in hist_files:
        with open(hf_path, 'rb') as f:
            data = pickle.load(f)
        filename, file_extension = os.path.splitext(hf)
        save_history_plots(data, os.path.join(tmp_dir, filename+".png"))
        image_insert(filename)
