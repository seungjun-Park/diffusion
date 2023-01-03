import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from util import partial, get_module

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, config=None):
        assert not config is None
        super().__init__()

        self.batch_size = config.batch_size
        self.dataset_configs = dict()
        self.num_workers = config.num_workers if config.num_workers is not None else self.batch_size * 2
        self.use_worker_init_fn = config.use_worker_init_fn

        if config.train is not None:
            self.dataset_configs["train"] = config.train
            self.train_dataloader = partial(self._train_dataloader, shuffle=config.train.shuffle)
        if config.validation is not None:
            self.dataset_configs["validation"] = config.validation
            self.val_dataloader = partial(self._validation_dataloader, shuffle=config.validation.shuffle)
        if config.test is not None:
            self.dataset_configs["test"] = config.test
            self.test_dataloader = partial(self._test_dataloader, shuffle=config.test.shuffle)
        if config.predict is not None:
            self.dataset_configs["predict"] = config.predict
            self.predict_dataloader = self._predict_dataloader

        self.wrap = config.wrap

    def prepare_data(self) -> None:
        NotImplemented()

    def setup(self, stage: str) -> None:
        self.datasets = dict((k, get_module(self.dataset_configs[k]) for k in self.dataset_configs))

        if self.wrap:
            for k in self.datasets:
                self.dataset_configs[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self, shuffle=True):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle, worker_init_fn=None)

    def _validation_dataloader(self, shuffle=True):
        return DataLoader(self.datasets['validation'], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle, worker_init_fn=None)

    def _test_dataloader(self, shuffle=True):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle, worker_init_fn=None)

    def _predict_dataloader(self, shuffle=True):
        return DataLoader(self.datasets['predict'], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=shuffle, worker_init_fn=None)
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]