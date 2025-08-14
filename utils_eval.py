"""
modified from pytorch-ood library
https://github.com/kkirchheim/pytorch-ood

@InProceedings{kirchheim2022pytorch,
    author    = {Kirchheim, Konstantin and Filax, Marco and Ortmeier, Frank},
    title     = {PyTorch-OOD: A Library for Out-of-Distribution Detection Based on PyTorch},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {4351-4360}
}


"""


from pytorch_ood.benchmark import Benchmark as bench_ood
from pytorch_ood.api import Detector
from pytorch_ood.dataset.img import LSUNCrop, LSUNResize, TinyImageNetCrop, TinyImageNetResize, Textures, Places365 
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from pytorch_ood.utils import OODMetrics, ToUnknown, ToRGB
from torchvision.transforms import Compose
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset



class CIFAR10_ood(bench_ood):

    def __init__(self, root, transform):

        self.transform = transform
        self.transform2 = Compose([ToRGB(), transform])
        self.train_in = CIFAR10(root, download=True, transform=transform, train=True)
        self.test_in = CIFAR10(root, download=True, transform=transform, train=False)

        self.test_oods = [
            CIFAR100(
                root, download=True, transform=self.transform, target_transform=ToUnknown(), train=False
            ),
            MNIST(
                root, download=True, transform=self.transform2, target_transform=ToUnknown(), train=False
            ),
            FashionMNIST(
                root, download=True, transform=self.transform2, target_transform=ToUnknown(), train=False
            ),
            Textures(
                root, download=True, transform=self.transform2, target_transform=ToUnknown()
            ),
            Places365(
                root, download=True, transform=self.transform2, target_transform=ToUnknown()
            ),
            TinyImageNetCrop(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
            TinyImageNetResize(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
            LSUNCrop(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
            LSUNResize(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
       
        ]

        self.ood_names: List[str] = []  #: OOD Dataset names
        self.ood_names = [type(d).__name__ for d in self.test_oods]

    def train_set(self) -> Dataset:

        return self.train_in

    def test_sets(self, known=True, unknown=True) -> List[Dataset]:


        if known and unknown:
            return [self.test_in + other for other in self.test_oods]

        if known and not unknown:
            return [self.train_in]

        if not known and unknown:
            return self.test_oods

        raise ValueError()

    def evaluate(
            self, detector: Detector, loader_kwargs: Dict = None, device: str = "cpu"
    ) -> List[Dict]:

        if loader_kwargs is None:
            loader_kwargs = {}

        metrics = []

        for name, dataset in zip(self.ood_names, self.test_sets()):
            loader = DataLoader(dataset=dataset, **loader_kwargs)
            print(name)
            m = OODMetrics()

            for x, y in loader:
                m.update(detector(x.to(device)), y)

            r = m.compute()
            r.update({"Dataset": name})

            metrics.append(r)

        return metrics
    

class CIFAR100_ood(bench_ood):


    def __init__(self, root, transform):

        self.transform = transform
        self.transform2 = Compose([ToRGB(), transform])
        self.train_in = CIFAR100(root, download=True, transform=transform, train=True)
        self.test_in = CIFAR100(root, download=True, transform=transform, train=False)

        self.test_oods = [
            CIFAR10(
                root, download=True, transform=self.transform, target_transform=ToUnknown(), train=False
            ),
            MNIST(
                root, download=True, transform=self.transform2, target_transform=ToUnknown(), train=False
            ),
            FashionMNIST(
                root, download=True, transform=self.transform2, target_transform=ToUnknown(), train=False
            ),
            Textures(
                root, download=True, transform=self.transform2, target_transform=ToUnknown()
            ),
            Places365(
                root, download=True, transform=self.transform2, target_transform=ToUnknown()
            ),
            TinyImageNetCrop(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
            TinyImageNetResize(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
            LSUNCrop(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
            LSUNResize(
                root, download=True, transform=transform, target_transform=ToUnknown()
            ),
       
        ]

        self.ood_names: List[str] = []  #: OOD Dataset names
        self.ood_names = [type(d).__name__ for d in self.test_oods]

    def train_set(self) -> Dataset:

        return self.train_in

    def test_sets(self, known=True, unknown=True) -> List[Dataset]:


        if known and unknown:
            return [self.test_in + other for other in self.test_oods]

        if known and not unknown:
            return [self.train_in]

        if not known and unknown:
            return self.test_oods

        raise ValueError()

    def evaluate(
            self, detector: Detector, loader_kwargs: Dict = None, device: str = "cpu"
    ) -> List[Dict]:

        if loader_kwargs is None:
            loader_kwargs = {}

        metrics = []

        for name, dataset in zip(self.ood_names, self.test_sets()):
            loader = DataLoader(dataset=dataset, **loader_kwargs)
            print(name)
            m = OODMetrics()

            for x, y in loader:
                m.update(detector(x.to(device)), y)

            r = m.compute()
            r.update({"Dataset": name})

            metrics.append(r)

        return metrics