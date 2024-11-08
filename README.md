# tagger_embedding_aligner

Tagger embedding aligner for smilingwolf's taggers

## Installation

```shell
git clone https://github.com/deepghs/tagger_embedding_aligner.git
cd tagger_embedding_aligner
pip install -r requiremnets.txt
```

## Train a model

```python
from ditk import logging
from torch.utils.data import Dataset

from aligner.dataset import EmbeddingDataset, get_block_file
from aligner.split import dataset_split
from aligner.train import train_model

logging.try_init_root(level=logging.INFO)

tagger_name = 'SwinV2_v3'  # train an embedding aligner for this tagger model
model_name = 'simple_num'  # select a model architecture, see aligner/models.py

dataset: Dataset = EmbeddingDataset(
    # datasets are stored in https://huggingface.co/datasets/deepghs/wd14_tagger_inversion
    # sample files has 200/2000/20000 samples ones, can be used for quick experiments
    # each block file has 100k+ samples (each approx 5GB), use them to train production models
    # make sure you have enough system RAM, this can be really huge
    npz_files=[
        # get_samples_file(model_name=tagger_name, samples=20000),
        get_block_file(model_name=tagger_name, block_id=i) for i in range(10)  # max to 60+
    ]
)
test_ratio = 0.2
train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])

if __name__ == '__main__':
    train_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        suffix_model_name=tagger_name,
        model_name=model_name,
        workdir=f'runs/{tagger_name}_m_{model_name}_my_custom_suffix',
        batch_size=16,
        learning_rate=1e-3,
        num_workers=16,
        max_epochs=100,
        weight_decay=1e-3,
    )

```

## Add Your Custom Model

Add a new one in `aligner/model.py`, and remember to register it with `register_converter` function.

## Publish Your Model

```shell
# export HF_TOKEN=xxxxx

python -m aligner.publish -r your/huggingface_repo -w runs/tagger_name_m_model_name_my_custom_suffix [-n model_name]
```

## Make List For Your Repository

```shell
# export HF_TOKEN=xxxxx

python -m aligner.list -r your/huggingface_repo
```

