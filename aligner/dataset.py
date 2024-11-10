import logging
from collections.abc import Sized
from typing import List

import numpy as np
from hbutils.random import keep_global_state, global_seed
from hbutils.string import plural_word
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from torch.utils.data import random_split
from tqdm import tqdm

_REPO_ID = 'deepghs/wd14_tagger_inversion'


def get_samples_file(model_name: str = 'SwinV2_v3', samples: int = 2000):
    return hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='dataset',
        filename=f'{model_name}/samples_{samples}.npz'
    )


def get_block_file(model_name: str = 'SwinV2_v3', block_id: int = 0):
    return hf_hub_download(
        repo_id=_REPO_ID,
        repo_type='dataset',
        filename=f'{model_name}/training_data_part{block_id}.npz'
    )


class EmbeddingDataset(Dataset, Sized):
    def __init__(self, npz_files: List[str]):
        Dataset.__init__(self)
        self.npz_files = npz_files

        self._pairs, self._lengths = [], []
        for npz_file in tqdm(self.npz_files, desc='Loading Emb Files'):
            _weight = np.load(npz_file)
            embs, preds = _weight['embs'], _weight['preds']
            self._pairs.append((embs, preds))
            self._lengths.append(embs.shape[0])

        self._lengths = np.array(self._lengths)
        self._prefixes = np.cumsum(self._lengths)
        logging.info(f'{plural_word(self._prefixes[-1].item(), "sample")} in total found in {self!r}.')

    def __getitem__(self, item):
        file_idx = np.searchsorted(self._prefixes, item, side='right')
        if file_idx == 0:
            idx_in_file = item
        else:
            idx_in_file = item - self._prefixes[file_idx - 1]

        # print(file_idx, idx_in_file)
        l_emb, l_pred = self._pairs[file_idx]
        embedding = l_emb[idx_in_file]
        norm_value = np.linalg.norm(embedding)
        embedding /= norm_value
        prediction = l_pred[idx_in_file]

        return embedding, prediction, norm_value

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
            get_samples_file(samples=2000),
            get_samples_file(samples=20000),
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
