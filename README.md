# tagger_embedding_aligner

Tagger embedding aligner for smilingwolf's taggers

## Why We Have To Do This

Smilingwolf's tagger generates embeddings for given images, which are then processed through a fully connected layer and
sigmoid to produce inference results. These embeddings are highly practical, prompting models like SigLIP to use them
for semantic alignment, mapping data from different modalities into the same cosine space.

Corresponding images and other modal data are mapped to similar embeddings, albeit with different modes.
Directly using embeddings from other modalities in the tagger's fully connected layer and sigmoid leads
to significant issues. Thus, **this project aims to find an optimal model structure to DE-NORMALIZE embeddings of
any morm to match the distribution of the tagger's original embeddings,
allowing for reliable tagging results—enabling danbooru-like tagging operations for data mapped to
this cosine space from any modality.**

## Installation

```shell
git clone https://github.com/deepghs/tagger_embedding_aligner.git
cd tagger_embedding_aligner
pip install -r requiremnets.txt
```

## Train A Model

Our dataset consists of real embeddings and prediction data from all of Smilingwolf's tagger models on the full danbooru
dataset, stored at [deepghs/wd14_tagger_inversion](https://huggingface.co/datasets/deepghs/wd14_tagger_inversion). The
provided code includes pre-packaged scripts for loading this dataset, allowing you to use them seamlessly and
painlessly.

```python
from ditk import logging
from torch.utils.data import Dataset

from aligner.dataset import EmbeddingDataset, get_block_file, get_samples_file
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

The input of the models should be L2-normalized embeddings,
the output of the models should be de-normalized embeddings,
batches are supported, the inputs and the outputs should have the same shape.

We utilize a total of four metrics in our evaluation:

* **EMB Cosine (emb_cos)**: Measures the cosine similarity between the model's output embeddings and the real embeddings
  from the dataset, indicating the semantic alignment between generated and real embeddings. A low value suggests
  significant semantic drift in the model.
* **EMB Norm (emb_norms)**: Represents the absolute difference in norms between the model's output embeddings and the
  real embeddings from the dataset.
* **Pred Loss (loss)**: Calculates the BCE loss between the inference results derived from passing the model's output
  embeddings through the original tagger's fully connected layer and sigmoid, and the real tagger's inference results,
  used as the loss function in our experiment.
* **Pred MSE (pred_mse)**: Computes the MSE between the inference results obtained from the model's output embeddings
  through the original tagger's fully connected layer and sigmoid, and the real tagger's inference results, serving as
  the primary metric in our experiment.

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

