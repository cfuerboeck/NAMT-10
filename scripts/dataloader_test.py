import os

import matplotlib.pyplot as plt
from PIL import Image


def plot_image(img):
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.show()


from key2med.data.loader import StudiesDataLoader

dataloader = StudiesDataLoader(
    data_path="/data/MEDICAL/datasets/CheXpert/CheXpert-v1.0",
    batch_size=4,
    img_resize=224,
    splits="train_valid_test",
    channels=3,
    do_random_transform=True,
    use_cache=True,
    in_memory=False,
    max_size=10_000,
    plot_stats=False,
)

for batch in dataloader.train:
    batch_images, batch_labels, batch_metadata = batch
    for images, label, metadatas in zip(batch_images, batch_labels, batch_metadata):
        print(label)
        for image, metadata in zip(images, metadatas):
            plot_image(image)
            print(metadata)
    break
