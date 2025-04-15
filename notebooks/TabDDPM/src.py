# ENV.PY
"""
Have not used in TabDDPM project.
"""

import datetime
import os
import shutil
import typing as ty
from pathlib import Path

PROJ = Path('tab-ddpm/').absolute().resolve()
EXP = PROJ / 'exp'
DATA = PROJ / 'data'


def get_path(path: ty.Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_absolute():
        path = PROJ / path
    return path.resolve()


def get_relative_path(path: ty.Union[str, Path]) -> Path:
    return get_path(path).relative_to(PROJ)


def duplicate_path(
    src: ty.Union[str, Path], alternative_project_dir: ty.Union[str, Path]
) -> None:
    src = get_path(src)
    alternative_project_dir = get_path(alternative_project_dir)
    dst = alternative_project_dir / src.relative_to(PROJ)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst = dst.with_name(
            dst.name + '_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        )
    (shutil.copytree if src.is_dir() else shutil.copyfile)(src, dst)

# UTIL.PY
import argparse
import atexit
import enum
import json
import os
import pickle
import shutil
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, List, Dict, Type, Optional, Tuple, TypeVar, Union, cast, get_args, get_origin

import __main__
import numpy as np
import tomli
import tomli_w
import torch
import zero
import typing as ty

# from . import env

RawConfig = Dict[str, Any]
Report = Dict[str, Any]
T = TypeVar('T')


class Part(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __str__(self) -> str:
        return self.value


class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value


# class Timer(zero.Timer):
#     @classmethod
#     def launch(cls) -> 'Timer':
#         timer = cls()
#         timer.run()
#         return timer


def update_training_log(training_log, data, metrics):
    def _update(log_part, data_part):
        for k, v in data_part.items():
            if isinstance(v, dict):
                _update(log_part.setdefault(k, {}), v)
            elif isinstance(v, list):
                log_part.setdefault(k, []).extend(v)
            else:
                log_part.setdefault(k, []).append(v)

    _update(training_log, data)
    transposed_metrics = {}
    for part, part_metrics in metrics.items():
        for metric_name, value in part_metrics.items():
            transposed_metrics.setdefault(metric_name, {})[part] = value
    _update(training_log, transposed_metrics)


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = '__none__'


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        tomli_w.dump(pack_config(config), f)
    # check that there are no bugs in all these "pack/unpack" things
    assert config == load_config(path)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def load(path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](Path(path), **kwargs)


def dump(x: Any, path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'dump_{Path(path).suffix[1:]}'](x, Path(path), **kwargs)


def _get_output_item_path(
    path: Union[str, Path], filename: str, must_exist: bool
) -> Path:
    path = env.get_path(path)
    if path.suffix == '.toml':
        path = path.with_suffix('')
    if path.is_dir():
        path = path / filename
    else:
        assert path.name == filename
    assert path.parent.exists()
    if must_exist:
        assert path.exists()
    return path


def load_report(path: Path) -> Report:
    return load_json(_get_output_item_path(path, 'report.json', True))


def dump_report(report: dict, path: Path) -> None:
    dump_json(report, _get_output_item_path(path, 'report.json', False))


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    with np.load(_get_output_item_path(path, 'predictions.npz', True)) as predictions:
        return {x: predictions[x] for x in predictions}


def dump_predictions(predictions: Dict[str, np.ndarray], path: Path) -> None:
    np.savez(_get_output_item_path(path, 'predictions.npz', False), **predictions)


def dump_metrics(metrics: Dict[str, Any], path: Path) -> None:
    dump_json(metrics, _get_output_item_path(path, 'metrics.json', False))


def load_checkpoint(path: Path, *args, **kwargs) -> Dict[str, np.ndarray]:
    return torch.load(
        _get_output_item_path(path, 'checkpoint.pt', True), *args, **kwargs
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def _print_sep(c, size=100):
    print(c * size)


def start(
    config_cls: Type[T] = RawConfig,
    argv: Optional[List[str]] = None,
    patch_raw_config: Optional[Callable[[RawConfig], None]] = None,
) -> Tuple[T, Path, Report]:  # config  # output dir  # report
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--continue', action='store_true', dest='continue_')
    if argv is None:
        program = __main__.__file__
        args = parser.parse_args()
    else:
        program = argv[0]
        try:
            args = parser.parse_args(argv[1:])
        except Exception:
            print(
                'Failed to parse `argv`.'
                ' Remember that the first item of `argv` must be the path (relative to'
                ' the project root) to the script/notebook.'
            )
            raise
    args = parser.parse_args(argv)

    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert args.continue_

    config_path = env.get_path(args.config)
    output_dir = config_path.with_suffix('')
    _print_sep('=')
    print(f'[output] {output_dir}')
    _print_sep('=')

    assert config_path.exists()
    raw_config = load_config(config_path)
    if patch_raw_config is not None:
        patch_raw_config(raw_config)
    if is_dataclass(config_cls):
        config = from_dict(config_cls, raw_config)
        full_raw_config = asdict(config)
    else:
        assert config_cls is dict
        full_raw_config = config = raw_config
    full_raw_config = asdict(config)

    if output_dir.exists():
        if args.force:
            print('Removing the existing output and creating a new one...')
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        elif not args.continue_:
            backup_output(output_dir)
            print('The output directory already exists. Done!\n')
            sys.exit()
        elif output_dir.joinpath('DONE').exists():
            backup_output(output_dir)
            print('The "DONE" file already exists. Done!')
            sys.exit()
        else:
            print('Continuing with the existing output...')
    else:
        print('Creating the output...')
        output_dir.mkdir()

    report = {
        'program': str(env.get_relative_path(program)),
        'environment': {},
        'config': full_raw_config,
    }
    if torch.cuda.is_available():  # type: ignore[code]
        report['environment'].update(
            {
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpus': zero.hardware.get_gpus_info(),
                'torch.version.cuda': torch.version.cuda,
                'torch.backends.cudnn.version()': torch.backends.cudnn.version(),  # type: ignore[code]
                'torch.cuda.nccl.version()': torch.cuda.nccl.version(),  # type: ignore[code]
            }
        )
    dump_report(report, output_dir)
    dump_json(raw_config, output_dir / 'raw_config.json')
    _print_sep('-')
    pprint(full_raw_config, width=100)
    _print_sep('-')
    return cast(config_cls, config), output_dir, report


_LAST_SNAPSHOT_TIME = None


def backup_output(output_dir: Path) -> None:
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output_dir.relative_to(env.PROJ)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output_dir = dir_ / relative_output_dir
        prev_backup_output_dir = new_output_dir.with_name(new_output_dir.name + '_prev')
        new_output_dir.parent.mkdir(exist_ok=True, parents=True)
        if new_output_dir.exists():
            new_output_dir.rename(prev_backup_output_dir)
        shutil.copytree(output_dir, new_output_dir)
        # the case for evaluate.py which automatically creates configs
        if output_dir.with_suffix('.toml').exists():
            shutil.copyfile(
                output_dir.with_suffix('.toml'), new_output_dir.with_suffix('.toml')
            )
        if prev_backup_output_dir.exists():
            shutil.rmtree(prev_backup_output_dir)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def _get_scores(metrics: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, float]]:
    return (
        {k: v['score'] for k, v in metrics.items()}
        if 'score' in next(iter(metrics.values()))
        else None
    )


def format_scores(metrics: Dict[str, Dict[str, Any]]) -> str:
    return ' '.join(
        f"[{x}] {metrics[x]['score']:.3f}"
        for x in ['test', 'val', 'train']
        if x in metrics
    )


def finish(output_dir: Path, report: dict) -> None:
    print()
    _print_sep('=')

    metrics = report.get('metrics')
    if metrics is not None:
        scores = _get_scores(metrics)
        if scores is not None:
            dump_json(scores, output_dir / 'scores.json')
            print(format_scores(metrics))
            _print_sep('-')

    dump_report(report, output_dir)
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if json_output_path:
        try:
            key = str(output_dir.relative_to(env.PROJ))
        except ValueError:
            pass
        else:
            json_output_path = Path(json_output_path)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_json(output_dir / 'report.json')
            json_output_path.write_text(json.dumps(json_data, indent=4))
        shutil.copyfile(
            json_output_path,
            os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
        )

    output_dir.joinpath('DONE').touch()
    backup_output(output_dir)
    print(f'Done! | {report.get("time")} | {output_dir}')
    _print_sep('=')
    print()


def from_dict(datacls: Type[T], data: dict) -> T:
    assert is_dataclass(datacls)
    data = deepcopy(data)
    for field in fields(datacls):
        if field.name not in data:
            continue
        if is_dataclass(field.type):
            data[field.name] = from_dict(field.type, data[field.name])
        elif (
            get_origin(field.type) is Union
            and len(get_args(field.type)) == 2
            and get_args(field.type)[1] is type(None)
            and is_dataclass(get_args(field.type)[0])
        ):
            if data[field.name] is not None:
                data[field.name] = from_dict(get_args(field.type)[0], data[field.name])
    return datacls(**data)


def replace_factor_with_value(
    config: RawConfig,
    key: str,
    reference_value: int,
    bounds: Tuple[float, float],
) -> None:
    factor_key = key + '_factor'
    if factor_key not in config:
        assert key in config
    else:
        assert key not in config
        factor = config.pop(factor_key)
        assert bounds[0] <= factor <= bounds[1]
        config[key] = int(factor * reference_value)


def get_temporary_copy(path: Union[str, Path]) -> Path:
    path = env.get_path(path)
    assert not path.is_dir() and not path.is_symlink()
    tmp_path = path.with_name(
        path.stem + '___' + str(uuid.uuid4()).replace('-', '') + path.suffix
    )
    shutil.copyfile(path, tmp_path)
    atexit.register(lambda: tmp_path.unlink())
    return tmp_path


def get_python():
    python = Path('python3.9')
    return str(python) if python.exists() else 'python'

def get_catboost_config(real_data_path, is_cv=False):
    ds_name = Path(real_data_path).name
    C = load_json(f'tuned_models/catboost/{ds_name}_cv.json')
    return C

def get_categories(X_train_cat):
    return (
        None
        if X_train_cat is None
        else [
            len(set(X_train_cat[:, i]))
            for i in range(X_train_cat.shape[1])
        ]
    )

# METRICS.PY
import enum
from typing import Any, Optional, Tuple, Dict, Union, cast
from functools import partial

import numpy as np
import scipy.special
import sklearn.metrics as skm

# from . import util
# from .util import TaskType


class PredictionType(enum.Enum):
    LOGITS = 'logits'
    PROBS = 'probs'

class MetricsReport:
    def __init__(self, report: dict, task_type: TaskType):
        self._res = {k: {} for k in report.keys()}
        if task_type in (TaskType.BINCLASS, TaskType.MULTICLASS):
            self._metrics_names = ["acc", "f1"]
            for k in report.keys():
                self._res[k]["acc"] = report[k]["accuracy"]
                self._res[k]["f1"] = report[k]["macro avg"]["f1-score"]
                if task_type == TaskType.BINCLASS:
                    self._res[k]["roc_auc"] = report[k]["roc_auc"]
                    self._metrics_names.append("roc_auc")

        elif task_type == TaskType.REGRESSION:
            self._metrics_names = ["r2", "rmse"]
            for k in report.keys():
                self._res[k]["r2"] = report[k]["r2"]
                self._res[k]["rmse"] = report[k]["rmse"]
        else:
            raise "Unknown TaskType!"

    def get_splits_names(self) -> list[str]:
        return self._res.keys()

    def get_metrics_names(self) -> list[str]:
        return self._metrics_names

    def get_metric(self, split: str, metric: str) -> float:
        return self._res[split][metric]

    def get_val_score(self) -> float:
        return self._res["val"]["r2"] if "r2" in self._res["val"] else self._res["val"]["f1"]
    
    def get_test_score(self) -> float:
        return self._res["test"]["r2"] if "r2" in self._res["test"] else self._res["test"]["f1"]
    
    def print_metrics(self) -> None:
        res = {
            "val": {k: np.around(self._res["val"][k], 4) for k in self._res["val"]},
            "test": {k: np.around(self._res["test"][k], 4) for k in self._res["test"]}
        }
    
        print("*"*100)
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])

        return res

class SeedsMetricsReport:
    def __init__(self):
        self._reports = []

    def add_report(self, report: MetricsReport) -> None:
        self._reports.append(report)
    
    def get_mean_std(self) -> dict:
        res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                res[split][metric] = [x.get_metric(split, metric) for x in self._reports]

        agg_res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                for k, f in [("count", len), ("mean", np.mean), ("std", np.std)]:
                    agg_res[split][f"{metric}-{k}"] = f(res[split][metric])
        self._res = res
        self._agg_res = agg_res

        return agg_res

    def print_result(self) -> dict:
        res = {split: {k: float(np.around(self._agg_res[split][k], 4)) for k in self._agg_res[split]} for split in ["val", "test"]}
        print("="*100)
        print("EVAL RESULTS:")
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])
        print("="*100)
        return res

def calculate_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, std = None) -> float:
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: Optional[PredictionType]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(y_pred, axis=1)
        )
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        util.raise_unknown('prediction_type', prediction_type)

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype('int64'), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Union[str, TaskType],
    prediction_type: Optional[Union[str, PredictionType]],
    y_info: Dict[str, Any],
) -> Dict[str, Any]:
    # Example: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert 'std' in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info['std'])
        r2 = skm.r2_score(y_true, y_pred)
        result = {'rmse': rmse, 'r2': r2}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True)
        )
        if task_type == TaskType.BINCLASS:
            result['roc_auc'] = skm.roc_auc_score(y_true, probs)
    return result

# DEEP.PY
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast

# import rtdl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zero
from torch import Tensor

# from .util import TaskType


def cos_sin(x: Tensor) -> Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


@dataclass
class PeriodicOptions:
    n: int  # the output size is 2 * n
    sigma: float
    trainable: bool
    initialization: Literal['log-linear', 'normal']


class Periodic(nn.Module):
    def __init__(self, n_features: int, options: PeriodicOptions) -> None:
        super().__init__()
        if options.initialization == 'log-linear':
            coefficients = options.sigma ** (torch.arange(options.n) / options.n)
            coefficients = coefficients[None].repeat(n_features, 1)
        else:
            assert options.initialization == 'normal'
            coefficients = torch.normal(0.0, options.sigma, (n_features, options.n))
        if options.trainable:
            self.coefficients = nn.Parameter(coefficients)  # type: ignore[code]
        else:
            self.register_buffer('coefficients', coefficients)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        return cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_loss_fn(task_type: TaskType) -> Callable[..., Tensor]:
    return (
        F.binary_cross_entropy_with_logits
        if task_type == TaskType.BINCLASS
        else F.cross_entropy
        if task_type == TaskType.MULTICLASS
        else F.mse_loss
    )


def default_zero_weight_decay_condition(module_name, module, parameter_name, parameter):
    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            rtdl.CLSToken,
            rtdl.NumericalFeatureTokenizer,
            rtdl.CategoricalFeatureTokenizer,
            Periodic,
        ),
    )


def split_parameters_by_weight_decay(
    model: nn.Module, zero_weight_decay_condition=default_zero_weight_decay_condition
) -> list[dict[str, Any]]:
    parameters_info = {}
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters():
            full_parameter_name = (
                f'{module_name}.{parameter_name}' if module_name else parameter_name
            )
            parameters_info.setdefault(full_parameter_name, ([], parameter))[0].append(
                zero_weight_decay_condition(
                    module_name, module, parameter_name, parameter
                )
            )
    params_with_wd = {'params': []}
    params_without_wd = {'params': [], 'weight_decay': 0.0}
    for full_parameter_name, (results, parameter) in parameters_info.items():
        (params_without_wd if any(results) else params_with_wd)['params'].append(
            parameter
        )
    return [params_with_wd, params_without_wd]


def make_optimizer(
    config: dict[str, Any],
    parameter_groups,
) -> optim.Optimizer:
    if config['optimizer'] == 'FT-Transformer-default':
        return optim.AdamW(parameter_groups, lr=1e-4, weight_decay=1e-5)
    return getattr(optim, config['optimizer'])(
        parameter_groups,
        **{x: config[x] for x in ['lr', 'weight_decay', 'momentum'] if x in config},
    )


def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def is_oom_exception(err: RuntimeError) -> bool:
    return any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


def train_with_auto_virtual_batch(
    optimizer,
    loss_fn,
    step,
    batch,
    chunk_size: int,
) -> tuple[Tensor, int]:
    batch_size = len(batch)
    random_state = zero.random.get_state()
    loss = None
    while chunk_size != 0:
        try:
            zero.random.set_state(random_state)
            optimizer.zero_grad()
            if batch_size <= chunk_size:
                loss = loss_fn(*step(batch))
                loss.backward()
            else:
                loss = None
                for chunk in zero.iter_batches(batch, chunk_size):
                    chunk_loss = loss_fn(*step(chunk))
                    chunk_loss = chunk_loss * (len(chunk) / batch_size)
                    chunk_loss.backward()
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            chunk_size //= 2
        else:
            break
    if not chunk_size:
        raise RuntimeError('Not enough memory even for batch_size=1')
    optimizer.step()
    return cast(Tensor, loss), chunk_size


def process_epoch_losses(losses: list[Tensor]) -> tuple[list[float], float]:
    losses_ = torch.stack(losses).tolist()
    return losses_, statistics.mean(losses_)

# DATA.PY
import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from importlib.resources import path
from pathlib import Path
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing
import torch
import os
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# from . import env, util
# from .metrics import calculate_metrics as calculate_metrics_
# from .util import TaskType, load_json

ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_MISSING_VALUE = 'nan'
CAT_RARE_VALUE = '__rare__'
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']


class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> 'Dataset':
        dir_ = Path(dir_)
        splits = [k for k in ['train', 'test'] if dir_.joinpath(f'y_{k}.npy').exists()]

        def load(item) -> ArrayDict:
            return {
                x: cast(np.ndarray, np.load(dir_ / f'{item}_{x}.npy', allow_pickle=True))  # type: ignore[code]
                for x in splits
            }

        if Path(dir_ / 'info.json').exists():
            info = util.load_json(dir_ / 'info.json')
        else:
            info = None
        return Dataset(
            load('X_num') if dir_.joinpath('X_num_train.npy').exists() else None,
            load('X_cat') if dir_.joinpath('X_cat_train.npy').exists() else None,
            load('y'),
            {},
            TaskType(info['task_type']),
            info.get('n_classes'),
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num['train'].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat['train'].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics_(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = 'rmse'
            score_sign = -1
        else:
            score_key = 'accuracy'
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics['score'] = score_sign * part_metrics[score_key]
        return metrics

def change_val(dataset: Dataset, val_size: float = 0.2):
    # should be done before transformations

    y = np.concatenate([dataset.y['train'], dataset.y['val']], axis=0)

    ixs = np.arange(y.shape[0])
    if dataset.is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)

    dataset.y['train'] = y[train_ixs]
    dataset.y['val'] = y[val_ixs]

    if dataset.X_num is not None:
        X_num = np.concatenate([dataset.X_num['train'], dataset.X_num['val']], axis=0)
        dataset.X_num['train'] = X_num[train_ixs]
        dataset.X_num['val'] = X_num[val_ixs]

    if dataset.X_cat is not None:
        X_cat = np.concatenate([dataset.X_cat['train'], dataset.X_cat['val']], axis=0)
        dataset.X_cat['train'] = X_cat[train_ixs]
        dataset.X_cat['val'] = X_cat[val_ixs]

    return dataset

def num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:

    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        # assert policy is None
        print('No NaNs in numerical features, skipping')
        return dataset

    assert policy is not None
    if policy == 'drop-rows':
        valid_masks = {k: ~v.any(1) for k, v in nan_masks.items()}
        assert valid_masks[
            'test'
        ].all(), 'Cannot drop test rows, since this will affect the final metrics.'
        new_data = {}
        for data_name in ['X_num', 'X_cat', 'y']:
            data_dict = getattr(dataset, data_name)
            if data_dict is not None:
                new_data[data_name] = {
                    k: v[valid_masks[k]] for k, v in data_dict.items()
                }
        dataset = replace(dataset, **new_data)
    elif policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        assert util.raise_unknown('policy', policy)
    return dataset

def Identity_transform(X):
    return X

# Inspired by: https://github.com/yandex-research/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def normalize(
    X: ArrayDict, normalization: Normalization, seed: Optional[int], return_normalizer : bool = False
) -> ArrayDict:
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
        # noise = 1e-3
        # if noise > 0:
        #     assert seed is not None
        #     stds = np.std(X_train, axis=0, keepdims=True)
        #     noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
        #     X_train = X_train + noise_std * np.random.default_rng(seed).standard_normal(
        #         X_train.shape
        #     )
    elif normalization == 'identity':
        normalizer = sklearn.preprocessing.FunctionTransformer(Identity_transform, inverse_func=Identity_transform)
    else:
        util.raise_unknown('normalization', normalization)

    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    assert X is not None
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):  # type: ignore[code]
        if policy is None:
            X_new = X
        elif policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)  # type: ignore[code]
            imputer.fit(X['train'])
            X_new = {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        else:
            util.raise_unknown('categorical NaN policy', policy)
    else:
        assert policy is None
        X_new = X
    return X_new


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X['train']) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X['train'].shape[1]):
        counter = Counter(X['train'][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
    return_encoder : bool = False
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != 'counter':
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(X['train'])
        encoder = make_pipeline(oe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X['train'].max(axis=0)
        for part in X.keys():
            if part == 'train': continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode.

    elif encoding == 'one-hot':
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse=False, dtype=np.float32 # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}

    elif encoding == 'counter':
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(('loe', loe))
        encoder.fit(X['train'], y_train)
        X = {k: encoder.transform(v).astype('float32') for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X['train'], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
    else:
        util.raise_unknown('encoding', encoding)
    
    if return_encoder:
        return X, True, encoder # type: ignore[code]
    return (X, True)


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {'policy': policy}
    if policy is None:
        pass
    elif policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    else:
        util.raise_unknown('policy', policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    return_transforms: bool = False
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(
            str(transformations).encode('utf-8')
        ).hexdigest()
        transformations_str = '__'.join(map(str, astuple(transformations)))
        cache_path = (
            cache_dir / f'cache__{transformations_str}__{transformations_md5}.pickle'
        )
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                print(
                    f"Using cached features: {cache_dir.name + '/' + cache_path.name}"
                )
                return value
            else:
                raise RuntimeError(f'Hash collision for {cache_path}')
    else:
        cache_path = None

    if dataset.X_num is not None:
        dataset = num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None and transformations.normalization is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True
        )
        num_transform = num_transform
    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
   
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y['train'],
            transformations.seed,
            return_encoder=True
        )

        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

        
    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    if cache_path is not None:
        util.dump_pickle((transformations, dataset), cache_path)
    # if return_transforms:
        # return dataset, num_transform, cat_transform
    return dataset


def build_dataset(
    path: Union[str, Path],
    transformations: Transformations,
    cache: bool
) -> Dataset:
    path = Path(path)
    dataset = Dataset.from_dir(path)
    return transform_dataset(dataset, transformations, path if cache else None)


def prepare_tensors(
    dataset: Dataset, device: Union[str, torch.device]
) -> Tuple[Optional[TensorDict], Optional[TensorDict], TensorDict]:
    X_num, X_cat, Y = (
        None if x is None else {k: torch.as_tensor(v) for k, v in x.items()}
        for x in [dataset.X_num, dataset.X_cat, dataset.y]
    )
    if device.type != 'cpu':
        X_num, X_cat, Y = (
            None if x is None else {k: v.to(device) for k, v in x.items()}
            for x in [X_num, X_cat, Y]
        )
    assert X_num is not None
    assert Y is not None
    if not dataset.is_multiclass:
        Y = {k: v.float() for k, v in Y.items()}
    return X_num, X_cat, Y

###############
## DataLoader##
###############

class TabDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset : Dataset, split : Literal['train', 'val', 'test']
    ):
        super().__init__()
        
        self.X_num = torch.from_numpy(dataset.X_num[split]) if dataset.X_num is not None else None
        self.X_cat = torch.from_numpy(dataset.X_cat[split]) if dataset.X_cat is not None else None
        self.y = torch.from_numpy(dataset.y[split])

        assert self.y is not None
        assert self.X_num is not None or self.X_cat is not None 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        out_dict = {
            'y': self.y[idx].long() if self.y is not None else None,
        }

        x = np.empty((0,))
        if self.X_num is not None:
            x = self.X_num[idx]
        if self.X_cat is not None:
            x = torch.cat([x, self.X_cat[idx]], dim=0)
        return x.float(), out_dict

def prepare_dataloader(
    dataset : Dataset,
    split : str,
    batch_size: int,
):

    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=1,
    )
    while True:
        yield from loader

def prepare_torch_dataloader(
    dataset : Dataset,
    split : str,
    shuffle : bool,
    batch_size: int,
) -> torch.utils.data.DataLoader:

    torch_dataset = TabDataset(dataset, split)
    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return loader

def dataset_from_csv(paths : Dict[str, str], cat_features, target, T):
    assert 'train' in paths
    y = {}
    X_num = {}
    X_cat = {} if len(cat_features) else None
    for split in paths.keys():
        df = pd.read_csv(paths[split])
        y[split] = df[target].to_numpy().astype(float)
        if X_cat is not None:
            X_cat[split] = df[cat_features].to_numpy().astype(str)
        X_num[split] = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)

    dataset = Dataset(X_num, X_cat, y, {}, None, len(np.unique(y['train'])))
    return transform_dataset(dataset, T, None)

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def prepare_fast_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat:
        X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
    else:
        X = torch.from_numpy(np.concatenate([D.X_num[split]], axis=1)).float()
    dataloader = FastTensorDataLoader(X, batch_size=batch_size, shuffle=(split=='train'))
    while True:
        yield from dataloader

def prepare_fast_torch_dataloader(
    D : Dataset,
    split : str,
    batch_size: int
):
    if D.X_cat is not None:
        X = torch.from_numpy(np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()
    y = torch.from_numpy(D.y[split])
    dataloader = FastTensorDataLoader(X, y, batch_size=batch_size, shuffle=(split=='train'))
    return dataloader

def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:,col])
        dist = cdist(X_synth[:, col][:, np.newaxis].astype(float), uniq[:, np.newaxis].astype(float))
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth

def concat_features(D : Dataset):
    if D.X_num is None:
        assert D.X_cat is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_cat.items()}
    elif D.X_cat is None:
        assert D.X_num is not None
        X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_num.items()}
    else:
        X = {
            part: pd.concat(
                [
                    pd.DataFrame(D.X_num[part], columns=range(D.n_num_features)),
                    pd.DataFrame(
                        D.X_cat[part],
                        columns=range(D.n_num_features, D.n_features),
                    ),
                ],
                axis=1,
            )
            for part in D.y.keys()
        }

    return X

def concat_to_pd(X_num, X_cat, y):
    if X_num is None:
        return pd.concat([
            pd.DataFrame(X_cat, columns=list(range(X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    if X_cat is not None:
        return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(X_cat, columns=list(range(X_num.shape[1], X_num.shape[1] + X_cat.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)
    return pd.concat([
            pd.DataFrame(X_num, columns=list(range(X_num.shape[1]))),
            pd.DataFrame(y, columns=['y'])
        ], axis=1)

def read_pure_data(path, split='train'):
    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)
    X_num = None
    X_cat = None
    if os.path.exists(os.path.join(path, f'X_num_{split}.npy')):
        X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')):
        X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True)

    return X_num, X_cat, y

def read_changed_val(path, val_size=0.2):
    path = Path(path)
    X_num_train, X_cat_train, y_train = read_pure_data(path, 'train')
    X_num_val, X_cat_val, y_val = read_pure_data(path, 'val')
    is_regression = load_json(path / 'info.json')['task_type'] == 'regression'

    y = np.concatenate([y_train, y_val], axis=0)

    ixs = np.arange(y.shape[0])
    if is_regression:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777)
    else:
        train_ixs, val_ixs = train_test_split(ixs, test_size=val_size, random_state=777, stratify=y)
    y_train = y[train_ixs]
    y_val = y[val_ixs]

    if X_num_train is not None:
        X_num = np.concatenate([X_num_train, X_num_val], axis=0)
        X_num_train = X_num[train_ixs]
        X_num_val = X_num[val_ixs]

    if X_cat_train is not None:
        X_cat = np.concatenate([X_cat_train, X_cat_val], axis=0)
        X_cat_train = X_cat[train_ixs]
        X_cat_val = X_cat[val_ixs]
    
    return X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val

#############

def load_dataset_info(dataset_dir_name: str) -> Dict[str, Any]:
    path = Path("data/" + dataset_dir_name)
    info = util.load_json(path / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    info['n_features'] = info['n_num_features'] + info['n_cat_features']
    info['path'] = path
    return info