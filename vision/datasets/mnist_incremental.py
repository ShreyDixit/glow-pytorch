from collections import defaultdict
import torchvision
from torch import tensor
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class MNISTIncremental(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=2, train=True):
        self.num_classes = num_classes
        self.ds = torchvision.datasets.MNIST(
            root=root_dir, train=train, download=True, transform=transform
        )

        self.label_dict = defaultdict(list)

        for i, (_, label) in enumerate(self.ds):
            self.label_dict[label].append(i)

    def __len__(self):
        return sum([len(self.label_dict[c]) for c in range(self.num_classes)])

    def __getitem__(self, index):
        for i in range(self.num_classes):
            if index >= len(self.label_dict[i]):
                index -= len(self.label_dict[i])

            else:
                img, label = self.ds[self.label_dict[i][index]]
                break
        return {"x": img, "y_onehot": one_hot(tensor(label), self.num_classes)}

    def add_classes(self, n):
        self.num_classes += n
