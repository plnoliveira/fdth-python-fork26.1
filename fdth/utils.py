import pandas as pd

from typing import Literal

def deduce_fdt_kind(data: pd.Series) -> Literal["categorical", "numerical"]:
    """
    Determine whether data is categorical or numerical based on its type.
    
    Parameters
    ----------
    data : pd.Series
        The input data series to analyze.
    
    Returns
    -------
    Literal["categorical", "numerical"]
        "categorical" if data contains strings or categorical values,
        "numerical" otherwise.
    """
    if len(data) == 0:
        return "numerical"
    
    first_val = data.iloc[0] if len(data) > 0 else None
    is_categorical = (data.dtype == "object" or 
                     data.dtype.name == 'category' or 
                     isinstance(first_val, str))
    return "categorical" if is_categorical else "numerical"
