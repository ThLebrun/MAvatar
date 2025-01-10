# type: ignore
from dataclasses import dataclass
from io import StringIO
from typing import Any, Type, Union

import msgspec
import numpy as np
import pandas as pd
from toolz.dicttoolz import keymap, valmap

from saiph.models import Model


@dataclass
class SerializedModel:
    data: Model
    __version__: str


class ModelJSONSerializer:
    # !Make sure to update the version if you change NumpyPandasEncoder or ModelJSONSerializer
    VERSION = "2.0"

    @classmethod
    def dumps(self, model: Model) -> bytes:
        to_encode = SerializedModel(data=model, __version__=self.VERSION)
        encoder = msgspec.json.Encoder(enc_hook=numpy_pandas_json_encoding_hook)
        encoded_model = encoder.encode(to_encode)
        return encoded_model

    @classmethod
    def loads(self, raw_model: Union[str, bytes]) -> Model:
        decoder = msgspec.json.Decoder(
            SerializedModel, dec_hook=numpy_pandas_json_decoding_hook
        )
        serialized_model = decoder.decode(raw_model)
        version = serialized_model.__version__
        major_version = version.split(".")[0]
        current_major_version = self.VERSION.split(".")[0]
        if major_version != current_major_version:
            raise ValueError(
                f"""Saved Model JSON version ({version}) is incompatible """
                f"""with the current serializer version ({self.VERSION})."""
            )

        return serialized_model.data


def numpy_pandas_json_encoding_hook(obj: Any) -> Any:
    """Encode numpy arrays, pandas dataframes, and pandas series, or objects containing them.

    Parameters:
        obj: object to encode
    """
    if isinstance(obj, np.ndarray):
        data = obj.tolist()
        return dict(__ndarray__=data, dtype=str(obj.dtype), shape=obj.shape)

    if isinstance(obj, pd.Series):
        data = obj.to_json(orient="index", default_handler=str)

        # We cast to string and not to object, to transform the
        # 'inferred_type' property (read-only) to string
        # If we don't it remains an integer when using an index with `'0'`
        index_type = "str" if str(obj.index.dtype) == "object" else str(obj.index.dtype)

        return dict(__series__=data, dtype=str(obj.dtype), index_type=index_type)

    if isinstance(obj, pd.DataFrame):
        # orient='table' includes dtypes but doesn't work
        dtypes = valmap(str, obj.dtypes.to_dict())

        # value_types is the type of a single cell in a column,
        # which can be different from dtypes, especially when the dtype is 'object'
        # or when we have nan-values together with integer values or stringified integers
        value_types = {}
        for col in obj.columns:
            non_null_series = obj[col].dropna()
            value_types[col] = (
                type(non_null_series.values[0]).__name__
                if not non_null_series.empty
                else "float"
            )
        data = obj.to_json(orient="index", default_handler=str)

        # We cast to string and not to object, to transform the
        # 'inferred_type' property (read-only) to string
        # If we don't it remains at integer when decoding and using
        # an index or column with `'0'`
        index_type = "str" if str(obj.index.dtype) == "object" else str(obj.index.dtype)
        columns_type = (
            "str" if str(obj.columns.dtype) == "object" else str(obj.columns.dtype)
        )

        return dict(
            __frame__=data,
            dtypes=dtypes,
            value_types=value_types,
            index_type=index_type,
            columns_type=columns_type,
        )

    raise NotImplementedError(f"Objects of type {type(obj)} cannot be encoded.")


def numpy_pandas_json_decoding_hook(type: Type, json_dict: Any) -> Any:
    """Decode numpy arrays, pandas dataframes, and pandas series, or objects containing them.

    Parameters:
        json_dict: json encoded object
    """
    # Numpy arrays
    if "__ndarray__" in json_dict:
        data = json_dict["__ndarray__"]
        return np.asarray(data, dtype=json_dict["dtype"]).reshape(json_dict["shape"])

    # Pandas Series
    if "__series__" in json_dict:
        data = json_dict["__series__"]
        series = pd.read_json(
            StringIO(data), orient="index", typ="series", dtype=json_dict["dtype"]
        )
        series.index = series.index.astype(json_dict["index_type"])
        return series

    # Pandas dataframes
    if "__frame__" in json_dict:
        data = json_dict["__frame__"]
        value_types = json_dict["value_types"]
        dtypes = json_dict["dtypes"]
        df = pd.read_json(StringIO(data), orient="index", typ="frame")

        df.index = df.index.astype(json_dict["index_type"])
        df.columns = df.columns.astype(json_dict["columns_type"])

        if json_dict["columns_type"] == "int64":
            # Integer columns names are saved as string in the json,
            # and pandas treats the name `'0'` differently that `0`.
            dtypes = keymap(int, json_dict["dtypes"])
            value_types = keymap(int, json_dict["value_types"])

        df = df.astype(value_types)
        df = df.astype(dtypes)

        return df

    return json_dict
