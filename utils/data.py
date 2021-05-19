import json
import os
import sys
import subprocess
import multiprocessing
from subprocess import CalledProcessError, TimeoutExpired
from typing import List
from tqdm import tqdm

import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs


class ASTDataset(DGLDataset):
    def __getitem__(self, idx):
        return self.graphs[idx], self.label_dict["label"][idx]  # because the labels start from 1

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


def cpp_compile_check(code: List, save_dir: str, idx: str):
    header = ["#include <iostream>", "#include <cstdio>", "#include <cstring>", "#include<algorithm>",
              "using namespace std;"]

    code = header + code
    save_path = os.path.join(save_dir, str(idx) + ".cpp")
    exe_path = os.path.join(save_dir, str(idx) + ".out")

    with open(save_path, "w", encoding='latin-1') as file:
        file.write("\n".join(code))

    try:
        subprocess.run(["g++", save_path, "-o", exe_path], capture_output=True, timeout=10, check=True)
    except CalledProcessError:
        return -1
    except TimeoutExpired:
        return -2
    return 0


def cpp_compile_check_interface(args):
    code, save_dir, idx = args
    return idx, cpp_compile_check(code, save_dir, idx)


def poj_104_check(json_file_path, mode, save_dir):
    with open(json_file_path, "r", encoding="latin-1") as file:
        lines = [json.loads(_l)['code'].replace("\t", " ").strip().split("\n") for _l in file.readlines()]

    tmp_dir = os.path.join(save_dir, "tmp")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    records = [(code, tmp_dir, idx) for idx, code in enumerate(lines)]

    pool = multiprocessing.Pool(processes=32)
    results = list((tqdm(pool.imap(cpp_compile_check_interface, records), total=len(records), desc='Check Bugs...')))
    results = [result for result in results if result is not None]
    pool.close()
    pool.join()

    results = json.dumps([idx for idx, status in results if status != 0])
    with open(os.path.join(save_dir, mode + "ce.txt"), "w") as file:
        file.write(results)


def main():
    json_file_path = sys.argv[1]
    mode = sys.argv[2]
    save_dir = sys.argv[3]
    poj_104_check(json_file_path, mode, save_dir)


if __name__ == "__main__":
    main()
