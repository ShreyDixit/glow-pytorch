import torchvision
from torch import tensor
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class MNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=10):
        self.num_classes = num_classes
        self.ds = torchvision.datasets.MNIST(
            root=root_dir, train=True, download=True, transform=transform
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        return {'x': img, 'y_onehot': one_hot(tensor(label), self.num_classes)}
