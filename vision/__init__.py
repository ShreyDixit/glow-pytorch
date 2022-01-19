from .datasets import CelebADataset, MNISTDataset, MNISTIncremental

Datasets = {
    "celeba": CelebADataset,
    "mnist": MNISTDataset,
    "mnist_incremental": MNISTIncremental,
}
