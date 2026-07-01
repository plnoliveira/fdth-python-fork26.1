from __future__ import annotations

from typing import Optional, Callable
from dataclasses import dataclass

import pandas as pd
import numpy as np

class Binning:
    """
    Class for managing binning information for numerical frequency distributions.
    
    Attributes
    ----------
    start : float
        Starting value of the first bin.
    end : float
        Ending value of the last bin.
    h : float
        Bin width (class interval size).
    k : int
        Number of bins.
    bins : np.ndarray
        Array of bin edges with length k+1.
    """
    
    def __init__(self, start: float, end: float, h: float, k: int, bins: np.ndarray):
        self.start = start
        self.end = end
        self.h = h
        self.k = k
        self.bins = bins
    
    @staticmethod
    def auto(
        start: Optional[float] = None,
        end: Optional[float] = None,
        h: Optional[float] = None,
        k: Optional[int] = None,
    ) -> Callable[[pd.Series], 'Binning']:
        """
        Create an automatic binning function based on specified parameters.
        
        Parameters
        ----------
        start : float, optional
            Starting value of the first bin.
        end : float, optional
            Ending value of the last bin.
        h : float, optional
            Bin width.
        k : int, optional
            Number of bins.
        
        Returns
        -------
        Callable[[pd.Series], 'Binning']
            A function that takes data and returns a Binning object.
        
        Raises
        ------
        ValueError
            If an invalid combination of parameters is provided.
        """
        
        def inner(data, start, end, h, k) -> 'Binning':
            all_none = all(x is None for x in [start, end, h, k])
            no_none = all(x is not None for x in [start, end])
            
            if all_none:
                return Binning.from_sturges(data)
            elif h is None and k is not None:
                return Binning.linspace(data=data, k=k)
            elif no_none and all(x is None for x in [h, k]):
                r = end - start
                k = int(np.ceil(np.sqrt(abs(r))))
                return Binning.linspace(k=max(k, 5), start=start, end=end)
            elif all(x is not None for x in [start, end, h]) and k is None:
                k = int(np.ceil((end - start) / h))
                return Binning.linspace(k=k, start=start, end=end)
            else:
                raise ValueError("Invalid combination of parameters")
        
        return lambda data: inner(data, start, end, h, k)
    
    @staticmethod
    def linspace(
        k: int,
        data: Optional[pd.Series] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> 'Binning':
        """
        Create linear binning with equal-width bins.
        
        Parameters
        ----------
        k : int
            Number of bins.
        data : pd.Series, optional
            Data for determining range if start/end not specified.
        start : float, optional
            Starting value of the first bin.
        end : float, optional
            Ending value of the last bin.
        
        Returns
        -------
        Binning
            A Binning object with linearly spaced bins.
        
        Raises
        ------
        ValueError
            If data is None when start or end is not specified.
        """
        if start is None:
            if data is None:
                raise ValueError("`data` is None when `start` was not specified")
            start = data.min() - abs(data.min()) / 100
        if end is None:
            if data is None:
                raise ValueError("`data` is None when `end` was not specified")
            end = data.max() + abs(data.max()) / 100
        
        h = (end - start) / k
        bins = np.linspace(start, end, k + 1)
        return Binning(k=k, start=start, end=end, h=h, bins=bins)
    
    @staticmethod
    def from_sturges(data: pd.Series) -> 'Binning':
        """
        Create binning using Sturges' rule.
        
        Parameters
        ----------
        data : pd.Series
            The data to bin.
        
        Returns
        -------
        Binning
            A Binning object with k = ceil(1 + log2(n)) bins.
        """
        n = len(data)
        if n == 0:
            return Binning.linspace(data=data, k=1)
        elif n == 1:
            return Binning.linspace(data=data, k=1)
        k = int(np.ceil(1 + np.log2(n)))
        return Binning.linspace(data=data, k=k)
    
    @staticmethod
    def from_scott(data: pd.Series) -> 'Binning':
        """
        Create binning using Scott's normal reference rule.
        
        Parameters
        ----------
        data : pd.Series
            The data to bin.
        
        Returns
        -------
        Binning
            A Binning object with bins sized for optimal normal distribution display.
        """
        n = len(data)
        if n == 0:
            return Binning.linspace(data=data, k=1)
        sd = np.std(data)
        at = data.max() - data.min()
        if sd == 0 or n == 1:
            return Binning.linspace(data=data, k=1)
        k = int(np.ceil(at / (3.5 * sd / (n ** (1 / 3)))))
        return Binning.linspace(data=data, k=max(k, 1))
    
    @staticmethod
    def from_fd(data: pd.Series) -> 'Binning':
        """
        Create binning using Freedman-Diaconis rule.
        
        Parameters
        ----------
        data : pd.Series
            The data to bin.
        
        Returns
        -------
        Binning
            A Binning object with bins based on IQR.
        """
        n = len(data)
        if n == 0:
            return Binning.linspace(data=data, k=1)
        elif n == 1:
            return Binning.linspace(data=data, k=1)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        if iqr == 0:
            return Binning.from_scott(data)
        k = int(np.ceil((data.max() - data.min()) / (2 * iqr / (n ** (1 / 3)))))
        return Binning.linspace(data=data, k=max(k, 1))
    
    def format_classes(self, round_: int, right: bool) -> list[str]:
        """
        Format class intervals for display.
        
        Parameters
        ----------
        round_ : int
            Number of decimal places for rounding.
        right : bool
            Whether intervals are right-closed (True) or left-closed (False).
        
        Returns
        -------
        list[str]
            Formatted class interval strings.
        """
        delims = ("(", "]") if right else ("[", ")")
        bins = self.bins
        
        def fmt(a, b) -> str:
            ra = str(round(a, round_))
            rb = str(round(b, round_))
            return f"{delims[0]}{ra}, {rb}{delims[1]}"
        
        return [fmt(a, b) for (a, b) in zip(bins[:-1], bins[1:])]
