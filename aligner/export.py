import glob
import json
import os
import re
import shutil
from typing import List, Optional

import torch
from ditk import logging
from hbutils.encoding import sha3

from .model import open_model_with_data
from .onnx import onnx_quick_export
from .profile import torch_model_profile

_LOG_FILE_PATTERN = re.compile(r'^events\.out\.tfevents\.(?P<timestamp>\d+)\.(?P<machine>[^.]+)\.(?P<extra>[\s\S]+)$')


def export_onnx(model, data, onnx_filename,
                input_names: List[Optional[str]] = None, output_names: List[Optional[str]] = None,
                opset_version: int = 14, verbose: bool = True, no_optimize: bool = False):
    logging.info('Preparing input and model ...')
    n = data['model_options']['n']
    example_input = torch.randn(1, n).float()
    model = model.cpu()
    model = model.float()
    model.eval()
    torch_model_profile(model, example_input)

    input_names = input_names or ['input']
    output_names = output_names or ['output']
    logging.info(f'Start exporting to {onnx_filename!r}')
    onnx_quick_export(
        model=model,
        example_input=example_input,
        onnx_filename=onnx_filename,
        opset_version=opset_version,
        verbose=verbose,
        no_optimize=no_optimize,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            **{name: {0: "batch"} for name in input_names},
            **{name: {0: "batch"} for name in output_names},
        },
        no_gpu=True,
    )


def export_to_directory(model_dir: str, dst_dir: str, logfile_anonymous: bool = True):
    model_file = os.path.join(model_dir, 'best.pt')
    os.makedirs(dst_dir, exist_ok=True)

    dst_model_file = os.path.join(dst_dir, 'model.pt')
    logging.info(f'Copying model {model_file!r} --> {dst_model_file!r}')
    shutil.copy(model_file, dst_model_file)
    model, data = open_model_with_data(model_file)

    n = data['model_options']['n']
    example_input = torch.randn(1, n).float()
    model = model.cpu()
    model = model.float()
    model.eval()
    flops, params = torch_model_profile(model, example_input)

    onnx_model_with_fc_file = os.path.join(dst_dir, 'model_with_fc.onnx')
    logging.info(f'Exporting {onnx_model_with_fc_file!r} ...')
    export_onnx(
        model=model,
        data=data,
        onnx_filename=onnx_model_with_fc_file,
        input_names=['input'],
        output_names=['prediction', 'embedding'],
    )

    onnx_model_file = os.path.join(dst_dir, 'model.onnx')
    logging.info(f'Exporting {onnx_model_file!r} ...')
    export_onnx(
        model=model.converter,
        data=data,
        onnx_filename=onnx_model_file,
        input_names=['input'],
        output_names=['embedding'],
    )

    metrics_file = os.path.join(dst_dir, 'metrics.json')
    logging.info(f'Writing metrics {metrics_file!r} ...')
    with open(metrics_file, 'w') as f:
        json.dump({
            'epoch': data['epoch'],
            **data['test_metrics'],
        }, f, ensure_ascii=False, sort_keys=True, indent=4)

    model_info_file = os.path.join(dst_dir, 'model.json')
    logging.info(f'Writing model info {model_info_file!r} ...')
    with open(model_info_file, 'w') as f:
        json.dump({
            'options': data['model_options'],
            'profile': {
                'flops': flops,
                'params': params,
            }
        }, f, ensure_ascii=False, sort_keys=True, indent=4)

    for logfile in glob.glob(os.path.join(model_dir, 'events.out.tfevents.*')):
        logging.info(f'Tensorboard file {logfile!r} found.')
        matching = _LOG_FILE_PATTERN.fullmatch(os.path.basename(logfile))
        assert matching, f'Log file {logfile!r}\'s name not match with pattern {_LOG_FILE_PATTERN.pattern}.'

        timestamp = matching.group('timestamp')
        machine = matching.group('machine')
        if logfile_anonymous:
            machine = sha3(machine.encode(), n=224)
        extra = matching.group('extra')

        final_name = f'events.out.tfevents.{timestamp}.{machine}.{extra}'
        shutil.copyfile(logfile, os.path.join(dst_dir, final_name))


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    export_to_directory(
        model_dir='runs/SwinV2_v3_m_simple_num_bs16_10blocks',
        dst_dir='test_export',
    )
