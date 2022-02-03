import logging
from functools import partial

from tqdm import tqdm

# make all tqdm pbars dynamic to fit any window resolution
tqdm = partial(tqdm, dynamic_ncols=True)
