import io
import json
import os
import re
from functools import partial

import click
import numpy as np
import pandas as pd
import torch
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from hfutils.utils import hf_fs_path
from huggingface_hub import HfFileSystem, HfApi
from huggingface_hub.constants import ENDPOINT
from huggingface_hub.hf_api import RepoFile
from tqdm.auto import tqdm

from .cli import GLOBAL_CONTEXT_SETTINGS
from .cli import print_version as _origin_print_version
from .model import open_model_with_data
from .profile import torch_model_profile
from .table import markdown_to_df

print_version = partial(_origin_print_version, 'list')


def _name_process(name: str):
    words = re.split(r'[\W_]+', name)
    return ' '.join([
        word.capitalize() if re.fullmatch('^[a-z0-9]+$', word) else word
        for word in words
    ])


_PERCENTAGE_METRICS = ('accuracy',)
HUGGINGFACE_CO_PAGE_TEMPLATE = ENDPOINT + "/{repo_id}/blob/{revision}/{filename}"


@click.command('list', context_settings={**GLOBAL_CONTEXT_SETTINGS},
               help='Publish model to huggingface model repository')
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def list_(repository: str, revision: str):
    logging.try_init_root(logging.INFO)
    hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))

    names = [fn.split('/')[-2] for fn in hf_fs.glob(f'{repository}@{revision}/*/model.pt')]
    logging.info(f'{plural_word(len(names), "model")} detected in {repository}@{revision}')

    rows = []
    for name in tqdm(names):
        item = {'Name': name}
        model, data = open_model_with_data(hf_client.hf_hub_download(
            repo_id=repository,
            filename=f'{name}/model.pt',
            revision=revision
        ))
        n = data['model_options']['n']
        item['Tagger'] = data['model_options']['suffix_model_name']
        item['Embedding Width'] = n
        input_ = torch.randn(1, n)
        flops, params = torch_model_profile(model, input_)
        with torch.no_grad():
            output, _ = model(input_)
        item['Tags Count'] = output.shape[-1]
        item['FLOPS'] = f'{flops / 1e9:.6f}G'
        item['Params'] = f'{params / 1e6:.2f}M'

        repo_file: RepoFile = list(hf_client.get_paths_info(
            repo_id=repository,
            repo_type='model',
            paths=[f'{name}/model.pt'],
            expand=True,
        ))[0]
        last_commit_at = repo_file.last_commit.date.timestamp()

        with open(hf_client.hf_hub_download(repository, f'{name}/metrics.json', revision=revision), 'r') as f:
            metrics = json.load(f)

        item['EMB Cosine'] = f"{metrics['emb_cos']:.4g}"
        item['EMB Norm'] = f"{metrics['emb_norms']:.4g}"
        item['Pred Loss'] = f"{metrics['loss']:.4g}"
        item['Pred MSE'] = f"{metrics['pred_mse']:.4g}"

        item['created_at'] = last_commit_at
        rows.append(item)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['created_at'], ascending=[False])
    del df['created_at']
    df = df.replace(np.nan, 'N/A')

    with TemporaryDirectory() as td:
        with open(os.path.join(td, 'README.md'), 'w') as f:
            if not hf_fs.exists(hf_fs_path(
                    repo_id=repository,
                    repo_type='model',
                    filename='README.md',
                    revision=revision,
            )):
                print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

            else:
                table_printed = False
                tb_lines = []
                with io.StringIO(hf_fs.read_text(hf_fs_path(
                        repo_id=repository,
                        repo_type='model',
                        filename='README.md',
                        revision=revision,
                )).rstrip() + os.linesep * 2) as ifx:
                    for line in ifx:
                        line = line.rstrip()
                        if line.startswith('|') and not table_printed:
                            tb_lines.append(line)
                        else:
                            if tb_lines:
                                df_c = markdown_to_df(os.linesep.join(tb_lines))
                                if 'Name' in df_c.columns and 'FLOPS' in df_c.columns and \
                                        'Params' in df_c.columns:
                                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)
                                    table_printed = True
                                    tb_lines.clear()
                                else:
                                    print(os.linesep.join(tb_lines), file=f)
                            print(line, file=f)

                if not table_printed:
                    print(df.to_markdown(index=False, numalign="center", stralign="center"), file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='model',
            revision=revision,
            path_in_repo='.',
            local_directory=td,
            message=f'Sync README for {repository}',
            hf_token=os.environ.get('HF_TOKEN'),
        )


if __name__ == '__main__':
    list_()
