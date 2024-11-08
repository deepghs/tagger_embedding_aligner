from collections.abc import Sized
from typing import List

import numpy as np
from hbutils.random import keep_global_state, global_seed
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from torch.utils.data import random_split

_REPO_ID = 'deepghs/wd14_tagger_inversion'


def _get_samples_file(model_name: str = 'SwinV2_v3', samples: int = 2000):
    return hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='dataset',
        filename=f'{model_name}/samples_{samples}.npz'
    )


def _get_block_file(model_name: str = 'SwinV2_v3', block_id: int = 0):
    return hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='dataset',
        filename=f'{model_name}/training_data_part{block_id}.npz'
    )


class EmbeddingDataset(Dataset, Sized):
    def __init__(self, npz_files: List[str]):
        Dataset.__init__(self)
        self.npz_files = npz_files
        self._weights = [np.load(x) for x in self.npz_files]
        self._lengths = np.array([x['embs'].shape[0] for x in self._weights])
        self._prefixes = np.cumsum(self._lengths)

    def __getitem__(self, item):
        file_idx = np.searchsorted(self._prefixes, item, side='right')
        if file_idx == 0:
            idx_in_file = item
        else:
            idx_in_file = item - self._prefixes[file_idx - 1]

        # print(file_idx, idx_in_file)
        data = self._weights[file_idx]
        embedding = data['embs'][idx_in_file]
        embedding /= np.linalg.norm(embedding)
        prediction = data['preds'][idx_in_file]

        return embedding, prediction

    def __len__(self):
        return self._prefixes[-1]


@keep_global_state()
def dataset_split(dataset, ratios: List[float], seed: int = 0):
    global_seed(seed)
    counts = (np.array(ratios) * len(dataset)).astype(int)
    counts[-1] = len(dataset) - counts[:-1].sum()
    assert counts.sum() == len(dataset)
    return random_split(dataset, counts)


if __name__ == '__main__':
    dataset = EmbeddingDataset(
        npz_files=[
            _get_samples_file(samples=2000),
            _get_samples_file(samples=20000),
        ]
    )
    print(len(dataset))

    emb, pred = dataset[0]
    print(emb.shape, pred.shape)
    emb, pred = dataset[1]
    print(emb.shape, pred.shape)

    emb, pred = dataset[1999]
    print(emb.shape, pred.shape)
    emb, pred = dataset[2000]
    print(emb.shape, pred.shape)

    print(np.linalg.norm(emb))
