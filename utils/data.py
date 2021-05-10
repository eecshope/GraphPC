import os

import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs


class ASTDataset(DGLDataset):
    def __getitem__(self, idx):
        return self.graphs[idx], self.label_dict["label"][idx] - 1  # because the labels start from 1

    def __len__(self):
        return len(self.graphs)

    def process(self):
        pass

    def __init__(self, data_type, raw_dir=None, save_dir=None):
        super(ASTDataset, self).__init__(name="ASTDataset", raw_dir=raw_dir, save_dir=save_dir)
        self.data_type = data_type
        self.graphs = list([])  # space for readable
        self.label_dict = {"labels": list([])}  # space for readable
        self.n_labels = 0

        if data_type not in ["train", "valid", "test"]:
            raise ValueError(f"Data type {data_type} is not available")

        if raw_dir is None and save_dir is None:
            raise ValueError("one of raw_dir and save_dir must be given")

        if raw_dir is not None and save_dir is None:
            raise ValueError("preprocess is not implemented")

        if save_dir is not None:
            self.load()

    def load(self):
        data_path = os.path.join(self.save_dir, self.data_type+".bin")
        self.graphs, self.label_dict = load_graphs(data_path)
        self.n_labels = torch.max(self.label_dict["label"]).item()
