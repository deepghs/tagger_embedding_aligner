import os.path
from typing import Optional

import click
from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, upload_directory_as_directory

from .cli import GLOBAL_CONTEXT_SETTINGS
from .export import export_to_directory


def publish_model(model_dir: str, repo_id: str, model_name: Optional[str] = None, revision: Optional[str] = None):
    hf_client = get_hf_client()

    model_name = model_name or os.path.basename(model_dir)
    if not hf_client.repo_exists(repo_id=repo_id, repo_type='model'):
        hf_client.create_repo(repo_id=repo_id, repo_type='model')

    with TemporaryDirectory() as td:
        export_to_directory(
            model_dir=model_dir,
            dst_dir=td,
        )

        upload_directory_as_directory(
            repo_id=repo_id,
            repo_type='model',
            local_directory=td,
            path_in_repo=model_name,
            revision=revision or 'main',
            message=f'Publish model {model_name!r}',
        )


@click.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS},
               help='Publish model to huggingface model repository')
@click.option('--workdir', '-w', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory of the training.', show_default=True)
@click.option('--name', '-n', 'name', type=str, default=None,
              help='Name of the checkpoint. Default is the basename of the work directory.', show_default=True)
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository for publishing model.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
def huggingface(workdir: str, name: Optional[str], repository: str, revision: str):
    logging.try_init_root(logging.INFO)

    publish_model(
        model_dir=workdir,
        model_name=name,
        repo_id=repository,
        revision=revision,
    )


if __name__ == '__main__':
    huggingface()
