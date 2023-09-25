import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .food101 import Food101
from .kinetics_sound import KineticsSound
from .vggsound import VggSound

class MultiModalDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 root_dir: str,
                 batch_size: int,
                 num_workers: int,
                 ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.pin_memory = True

        if dataset_name == "food101":
            self.train_dataset = Food101(mode="train", dataset_root_dir=self.root_dir)
            self.val_dataset = Food101(mode="test", dataset_root_dir=self.root_dir)
            self.test_dataset = Food101(mode="test", dataset_root_dir=self.root_dir)
        elif dataset_name == "kinetics_sound":
            self.train_dataset = KineticsSound(mode="train", dataset_root_dir=self.root_dir)
            self.val_dataset = KineticsSound(mode="test", dataset_root_dir=self.root_dir)
            self.test_dataset = KineticsSound(mode="test", dataset_root_dir=self.root_dir)
        elif dataset_name == "vggsound":
            self.train_dataset = VggSound(mode="train", dataset_root_dir=self.root_dir)
            self.val_dataset = VggSound(mode="test", dataset_root_dir=self.root_dir)
            self.test_dataset = VggSound(mode="test", dataset_root_dir=self.root_dir)
        else:
            raise NotImplemented(f"Unsupported dataset: {dataset_name}")

    @property
    def all_ids(self):
        return self.train_dataset.all_ids

    @property
    def unlabeled_ids(self):
        return self.train_dataset.unlabeled_ids

    @property
    def labeled_ids(self):
        return self.train_dataset.labeled_ids

    def query_for_label(self, queried_ids):
        self.train_dataset.query_for_label(queried_ids)

    def train_dataloader(self) -> DataLoader:
        self.train_dataset.train()
        assert self.train_dataset.mode == "train"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset.mode != "train"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset.mode != "train"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def unlabeled_dataloader(self, batch_size=None, num_workers=None) -> DataLoader:
        self.train_dataset.query()
        assert self.train_dataset.mode != "train"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers if num_workers is None else num_workers,
        )

    def splited_unlabeled_dataloader(self, split_size, split_index, batch_size=None, num_workers=None) -> DataLoader:
        self.train_dataset.query()
        num_of_data = len(self.train_dataset.unlabeled_ids)
        num_of_split_data = num_of_data // split_size
        if split_index == (split_size - 1):
            self.train_dataset.sample_ids = self.unlabeled_ids[num_of_split_data * split_index:]
        else:
            self.train_dataset.sample_ids = self.unlabeled_ids[num_of_split_data * split_index: num_of_split_data * (split_index + 1)]
        assert self.train_dataset.mode != "train"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers if num_workers is None else num_workers,
        )

    def whole_dataloader(self) -> DataLoader:
        self.train_dataset.sample_ids = self.train_dataset.all_ids
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )