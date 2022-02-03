import plotext as plx
import matplotlib.pyplot as plt
import matplotlib
import torchvision.utils
from collections import Counter
DEFAULT_DPI = matplotlib.rcParams['figure.dpi']


def image_to_tensor(x):
    return torchvision.transforms.ToTensor()(x)


def tensor_to_image(x):
    return torchvision.transforms.ToPILImage()(x)


def merge_alternating_lists(*lists):
    return [x for pair in zip(*lists) for x in pair]


def plot_image_grid(*tensors, as_image: bool = True):
    nrows = 8
    alternating_images = merge_alternating_lists(*[list(tensor) for tensor in tensors])
    grid = torchvision.utils.make_grid(alternating_images, nrow=nrows, padding=0)
    if as_image:
        return tensor_to_image(grid)
    return grid


def text_histogram(data, title: str = None, data_label: str = None):
    if isinstance(data[0], (int, float)):
        _text_histogram_numerical(data, title, data_label)
    elif isinstance(data[0], str):
        _text_histogram_categorical(data, title, data_label)
    else:
        raise NotImplementedError


def _text_histogram_numerical(data, title, data_label):
    plx.clp()
    plx.hist(data, bins=20, label=data_label or '')
    if title is not None:
        plx.title(title)
    plx.yscale('symlog')
    plx.xlabel("x")
    plx.ylabel("frequency")
    plx.plotsize(50, 20)
    plx.show()
    plx.clear_plot()


def _text_histogram_categorical(data, title, data_label):
    counter = Counter(data)
    keys = list(counter.keys())
    values = list(counter.values())
    plx.bar(keys, values)
    if title is not None:
        plx.title(title)
    plx.plotsize(50, 20)
    plx.yscale('symlog')
    plx.xlabel('x')
    plx.ylabel("frequency")
    plx.show()
    plx.clear_plot()
