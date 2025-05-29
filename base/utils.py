import csv
import sys
import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def param_count(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def display_loss(loss, save_path=None):
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def save_toFile(path, file_name, data_saved, rows=0):
    f = open(path + file_name, 'w')
    writer = csv.writer(f)
    if rows == 0:
        writer.writerow(data_saved)
    if rows == 1:
        writer.writerows(data_saved)
    f.close()



