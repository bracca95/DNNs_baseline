# inspired by quicktype.io

import os
import sys
import json

from dataclasses import dataclass
from typing import Optional, Union, List, Any, Callable, Iterable, Type, cast

from src.tools import Utils, Logger
from config.consts import *

def from_bool(x: Any) -> bool:
    Utils.check_instance(x, bool)
    return x

def from_int(x: Any) -> int:
    Utils.check_instance(x, int)
    return x

def from_float(x: Any) -> float:
    Utils.check_instance(x, float)
    return x

def from_str(x: Any) -> str:
    Utils.check_instance(x, str)
    return x

def from_none(x: Any) -> Any:
    Utils.check_instance(x, None)
    return x

def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    Utils.check_instance(x, list)
    return [f(y) for y in x]

def from_union(fs: Iterable[Any], x: Any):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    raise TypeError(f"{x} should be one out of {[type(f.__name__) for f in fs]}")


def to_class(c: Type[T], x: Any) -> dict:
    Utils.check_instance(x, c)
    return cast(Any, x).serialize()


@dataclass
class Config:
    train: Optional[bool] = None
    dataset_path: Optional[str] = None
    dataset_mean: Optional[List[float]] = None
    dataset_std: Optional[List[float]] = None
    batch_size: Optional[int] = 32
    epochs: Optional[int] = 100
    crop_size: Optional[int] = None
    image_size: Optional[int] = None
    defect_class: Optional[List[str]] = None

    @classmethod
    def deserialize(cls, str_path: str) -> 'Config':
        obj = Utils.read_json(str_path)
        
        try:
            train_tmp = from_union([from_str, from_bool, from_none], obj.get(CONFIG_TRAIN))
            train = Utils.str2bool(train_tmp) if isinstance(train_tmp, str) else train_tmp

            dataset_path = from_union([from_none, from_str], obj.get(CONFIG_DATASET_PATH))
            if dataset_path is None:
                dataset_path = input("insert dataset path: ")
            dataset_path = Utils.validate_path(dataset_path)

            dataset_mean = from_union([lambda x: from_list(from_float, x), from_none], obj.get(CONFIG_DATASET_MEAN))
            dataset_std = from_union([lambda x: from_list(from_float, x), from_none], obj.get(CONFIG_DATASET_STD))
            batch_size = from_union([from_none, from_int], obj.get(CONFIG_BATCH_SIZE))
            epochs = from_union([from_none, from_int], obj.get(CONFIG_EPOCHS))
            crop_size = from_union([from_none, from_int], obj.get(CONFIG_CROP_SIZE))
            image_size = from_union([from_none, from_int], obj.get(CONFIG_IMAGE_SIZE))
            defect_class = from_union([lambda x: from_list(from_str, x), from_none], obj.get(CONFIG_DEFECT_CLASS))
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)
        except FileNotFoundError as fnf:
            Logger.instance().critical(fnf.args)
            sys.exit(-1)
        
        Logger.instance().info(f"Config deserialized: " +
            f"train: {train}, dataset_path: {dataset_path}, batch_size {batch_size}, epochs: {epochs}" +
            f"dataset mean: {dataset_mean}, dataset_std: {dataset_std}, crop_size: {crop_size}, " +
            f"image_size: {image_size} defect_class: {defect_class}")
        
        return Config(train, dataset_path, dataset_mean, dataset_std, batch_size, crop_size, image_size, defect_class)

    def serialize(self, directory: str, filename: str):
        result: dict = {}
        dire = None

        try:
            dire = Utils.validate_path(directory)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"{fnf.args}")
            sys.exit(-1)
        
        # if you do not want to write null values, add a field to result if and only if self.field is not None
        result[CONFIG_TRAIN] = from_union([from_none, from_bool], self.train)
        result[CONFIG_DATASET_PATH] = from_union([from_none, from_str], self.dataset_path)
        result[CONFIG_DATASET_MEAN] = from_union([lambda x: from_list(from_float, x), from_none], self.dataset_mean)
        result[CONFIG_DATASET_STD] = from_union([lambda x: from_list(from_float, x), from_none], self.dataset_std)
        result[CONFIG_BATCH_SIZE] = from_union([from_none, from_int], self.batch_size)
        result[CONFIG_EPOCHS] = from_union([from_none, from_int], self.epochs)
        result[CONFIG_CROP_SIZE] = from_union([from_none, from_int], self.crop_size)
        result[CONFIG_IMAGE_SIZE] = from_union([from_none, from_int], self.image_size)
        result[CONFIG_DEFECT_CLASS] = from_union([lambda x: from_list(from_str, x), from_none], self.defect_class)

        with open(os.path.join(dire, filename), "w") as f:
            json_dict = json.dumps(result, indent=4)
            f.write(json_dict)

        Logger.instance().info("Config serialized")


def config_to_json(x: Config) -> Any:
    return to_class(Config, x)
