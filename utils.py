import matplotlib.pyplot as plt
import torch


def to_one_hot_vector(y):
    torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)


def show_img(idx, data_set):
    img = data_set.train_data[idx]
    plt.axis("off")
    plt.imshow(img, cmap="gray")
    plt.show()


def show_tensor(tensor):
    plt.axis("off")
    plt.imshow(tensor, cmap="gray")
    plt.show()