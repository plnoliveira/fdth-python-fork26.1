from typing import Literal, Optional, Any, Union

import pandas as pd
import numpy as np

from .utils import deduce_fdt_kind
from .numerical_fdt import NumericalFDT
from .categorical_fdt import CategoricalFDT
from .multiple_fdt import MultipleFDT

def fdt(
    data: Optional[Union[pd.Series, list, pd.DataFrame, np.ndarray, dict]] = None,
    *,
    freqs: Optional[Union[pd.Series, list, dict]] = None,
    kind: Literal["numerical", "categorical", None] = None,
    use_raw_data_stats: bool = False,
    by: Optional[Union[str, list[str]]] = None,
    **kwargs,
) -> Union[NumericalFDT, CategoricalFDT, MultipleFDT]:
    """
    Create frequency distribution table(s).
    
    Parameters
    ----------
    data : array-like, optional
        Input data.
    freqs : array-like or dict, optional
        Pre-calculated frequencies.
    kind : {"numerical", "categorical"}, optional
        Force data type.
    use_raw_data_stats : bool, default=False
        Use raw data for statistics.
    by : str or list of str, optional
        Column(s) to group by.
    **kwargs
        Additional arguments passed to FDT classes.
    
    Returns
    -------
    NumericalFDT, CategoricalFDT, or MultipleFDT
        Frequency distribution table(s).
    """
    
    if by is not None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Parameter 'by' can only be used with DataFrame input")
        
        kwargs['use_raw_data_stats'] = use_raw_data_stats
        return MultipleFDT(data, by=by, **kwargs)

    if isinstance(data, (pd.DataFrame, np.ndarray)):
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = pd.Series(data)
        else:
            kwargs['use_raw_data_stats'] = use_raw_data_stats
            return MultipleFDT(data, **kwargs)
    
    if isinstance(data, dict):
        df = pd.DataFrame(data)
        kwargs['use_raw_data_stats'] = use_raw_data_stats
        return MultipleFDT(df, **kwargs)
    
    if isinstance(data, (list, pd.Series)):
        data_ = pd.Series(data)
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data_ = pd.Series(data)
        else:
            data_ = pd.Series(data[:, 0])
    elif data is None and freqs is not None:
        if isinstance(freqs, (pd.Series, list)):
            if kind is not None and kind != "numerical":
                raise TypeError("`freqs` (as pandas.Series | list) can only be used with numerical FDTs")
            return NumericalFDT(freqs=freqs, use_raw_data_stats=False, **kwargs)
        elif isinstance(freqs, dict):
            if kind is not None and kind != "categorical":
                raise TypeError("`freqs` (as dict) can only be used with categorical FDTs")
            return CategoricalFDT(freqs=freqs, **kwargs)
        else:
            raise TypeError("`freqs` must be pandas.Series | list | dict when specified")
    else:
        raise TypeError("`data` must be list | pandas.Series | pandas.DataFrame | numpy.ndarray | dict")
    
    kind = kind or deduce_fdt_kind(data_)
    if kind == "categorical":
        return CategoricalFDT(data_, **kwargs)
    elif kind == "numerical":
        return NumericalFDT(data_, use_raw_data_stats=use_raw_data_stats, **kwargs)
    else:
        raise TypeError(f"Unexpected kind: {repr(kind)}")
