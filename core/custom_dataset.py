import torch
from torch.utils.data import TensorDataset
import json


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, class_dict_file):
        self.dataset = dataset
        self.class_relabler = self._set_class_relabler(class_dict_file)

    def __getitem__(self, index):
        data, target = self.dataset.dataset[index]
        return data, self.class_relabler(target), index

    def _set_class_relabler(self, class_dict_file):
        if not class_dict_file:
            return lambda x: x
        f = open(class_dict_file)
        class_dict_raw = json.load(f)
        class_dict = dict([(class_dict_raw[s][0], s) for s in class_dict_raw])
        relabler_dict = dict(
            [
                (self.dataset.dataset.class_to_idx[s], int(class_dict[s]))
                for s in self.dataset.dataset.class_to_idx
            ]
        )
        return lambda x: relabler_dict[x]

    def __len__(self):
        return len(self.dataset)
