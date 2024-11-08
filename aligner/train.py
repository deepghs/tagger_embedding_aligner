import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ditk import logging
from hbutils.random import global_seed
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import _get_samples_file, EmbeddingDataset, dataset_split
from .model import get_model
from .profile import torch_model_profile


def train_model(workdir: str, train_dataset: Dataset, test_dataset: Dataset,
                batch_size: int = 16, max_epochs: int = 100, eval_epoch: int = 1,
                num_workers: Optional[int] = None, learning_rate: float = 0.001, weight_decay: float = 1e-3,
                model_name: str = 'simple', suffix_model_name: str = 'SwinV2_v3',
                seed: int = 0, **options):
    if seed is not None:
        # native random, numpy, torch and faker's seeds are includes
        # if you need to register more library for seeding, see:
        # https://hansbug.github.io/hbutils/main/api_doc/random/state.html#register-random-source
        logging.info(f'Globally set the random seed {seed!r}.')
        global_seed(seed)

    tb_writer = SummaryWriter(workdir)

    accelerator = Accelerator(
        # mixed_precision=self.cfgs.mixed_precision,
        step_scheduler_with_optimizer=False,
    )

    weights = np.load(_get_samples_file(model_name=suffix_model_name, samples=200))
    embedding_width = weights['embs'].shape[-1]
    logging.info(f'Tagger model type: {suffix_model_name!r}, embedding width: {embedding_width!r}.')

    model: nn.Module = get_model(
        model_name=model_name,
        n=embedding_width,
        suffix_model_name=suffix_model_name,
        **options,
    )

    sample_input, _ = test_dataset[0]
    print(sample_input.shape)
    torch_model_profile(model, torch.tensor(sample_input).unsqueeze(0))  # profile the model

    num_workers = num_workers or min(os.cpu_count(), batch_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_dataloader), epochs=max_epochs,
        pct_start=0.15, final_div_factor=20.
    )

    # noinspection PyTypeChecker
    model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn = \
        accelerator.prepare(model, optimizer, train_dataloader, test_dataloader, scheduler, loss_fn)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss, train_sims = 0.0, 0.0
        train_total = 0
        for i, (input_embs, output_preds) in enumerate(tqdm(train_dataloader, desc=f'Train epoch #{epoch}')):
            input_embs = input_embs.float().to(accelerator.device)
            output_preds = output_preds.float().to(accelerator.device)

            optimizer.zero_grad()
            outputs, output_embs = model(input_embs)
            train_total += input_embs.shape[0]

            loss = loss_fn(outputs, output_preds)

            accelerator.backward(loss)
            optimizer.step()
            train_loss += loss.item() * input_embs.size(0)
            train_sims += F.cosine_similarity(input_embs, output_embs, dim=-1).sum()
            scheduler.step()

        train_loss /= train_total
        train_sims /= train_total
        logging.info(f'Train #{epoch}, loss: {train_loss:.6f}, emb cosine similarity: {train_sims}')
        tb_writer.add_scalar('train/loss', train_loss, epoch)

        if epoch % eval_epoch == 0:
            model.eval()
            test_loss, test_sims = 0.0, 0
            test_total = 0
            with torch.no_grad():
                for i, (input_embs, output_preds) in enumerate(tqdm(test_dataloader)):
                    input_embs = input_embs.float().to(accelerator.device)
                    output_preds = output_preds.to(accelerator.device)

                    outputs, output_embs = model(input_embs)
                    test_total += input_embs.shape[0]

                    loss = loss_fn(outputs, output_preds)
                    test_loss += loss.item() * input_embs.size(0)
                    test_sims += F.cosine_similarity(input_embs, output_embs, dim=-1).sum()

                test_loss /= test_total
                test_sims /= test_total
                logging.info(f'Test #{epoch}, loss: {test_loss:.6f}, emb cosine similarity: {train_sims}')
                tb_writer.add_scalar('test/loss', test_loss, epoch)


if __name__ == '__main__':
    logging.try_init_root(level=logging.INFO)
    tagger = 'SwinV2_v3'
    dataset: Dataset = EmbeddingDataset(
        npz_files=[
            _get_samples_file(model_name=tagger, samples=20000),
        ]
    )
    test_ratio = 0.2
    train_dataset, test_dataset = dataset_split(dataset, [1 - test_ratio, test_ratio])

    train_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        suffix_model_name=tagger,
        workdir=f'runs/{tagger}',
        batch_size=16,
    )
