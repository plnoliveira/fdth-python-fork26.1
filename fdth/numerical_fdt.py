from __future__ import annotations

from typing import Optional, Callable, Union
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .binning import Binning

class NumericalFDT:
    """
    Frequency Distribution Table for numerical data.
    
    Attributes
    ----------
    count : int
        Total number of observations.
    raw_data : pd.Series or None
        Original raw data if available.
    table : pd.DataFrame
        Frequency distribution table with columns:
        - Class limits: Bin intervals
        - f: Absolute frequency
        - rf: Relative frequency
        - rf(%): Relative frequency percentage
        - cf: Cumulative frequency
        - cf(%): Cumulative frequency percentage
    binning : Binning
        Binning information used for the table.
    use_raw_data_stats : bool
        Whether to use raw data for statistical calculations.
    """
    
    def __init__(
        self,
        data: Optional[Union[pd.Series, list]] = None,
        *,
        freqs: Optional[Union[pd.Series, list]] = None,
        binning: Optional[Union[Binning, Callable]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        h: Optional[float] = None,
        k: Optional[int] = None,
        right: bool = False,
        remove_nan: bool = False,
        round_: int = 2,
        use_raw_data_stats: bool = False,
    ):
        if (start is not None) or (end is not None) or (h is not None) or (k is not None):
            if binning is not None:
                raise ValueError("Cannot specify both `binning` and one of `start`/`end`/`h`/`k`")
            else:
                binning = Binning.auto(start=start, end=end, h=h, k=k)
        else:
            if binning is None:
                binning = Binning.from_sturges
        
        if data is not None and freqs is not None:
            raise ValueError("Exactly one of `data` or `freqs` must be specified")
        elif data is not None:
            self._column = data.name
            data = self._cleanup_data(data, remove_nan=remove_nan)
            self.count = len(data)
            
            if use_raw_data_stats:
                if hasattr(data, 'iloc') and isinstance(data, pd.Series):
                    self.raw_data = data.copy()
                else:
                    if data is not None:
                        import warnings
                        warnings.warn(
                            "use_raw_data_stats=True but only frequency data is available. "
                            "Falling back to midpoint approximation.",
                            UserWarning,
                            stacklevel=2
                        )
                    self.raw_data = None
                    use_raw_data_stats = False
            else:
                self.raw_data = None
            
            self.use_raw_data_stats = use_raw_data_stats
            
            b = binning(data) if callable(binning) else binning
            self.table = self._make_table_from_data(data, b, right, round_=round_)
            self.binning = b
        elif freqs is not None:
            if not isinstance(binning, Binning):
                raise ValueError("A ready-made binning must be specified when passing `freqs`")
            
            freqs = pd.Series(freqs)
            self.count = int(freqs.sum())
            self.raw_data = None
            self.use_raw_data_stats = False
            
            self.table = self._make_table_from_frequencies(freqs, binning, right, round_=round_)
            self.binning = binning
    
    @staticmethod
    def _cleanup_data(data: Union[pd.Series, list], remove_nan: bool) -> pd.Series:
        d = np.array([np.nan if v is None else v for v in data], dtype=np.float64)
        if not np.issubdtype(d.dtype, np.number):
            raise ValueError("Input data must be numeric")
        
        if remove_nan:
            d = d[~np.isnan(d)]
        elif np.any(np.isnan(d)):
            raise ValueError("The data has NaN values and remove_nan=False")
        
        return pd.Series(d)
    
    @lru_cache
    def _midpoints(self) -> np.ndarray:
        bins = self.binning.bins
        return 0.5 * (bins[:-1] + bins[1:])
    
    @lru_cache
    def mean(self) -> float:
        if self.use_raw_data_stats and self.raw_data is not None:
            return float(self.raw_data.mean())
        else:
            return np.sum(self.table["f"] * self._midpoints()) / self.count
    
    @lru_cache
    def median(self) -> float:
        if self.use_raw_data_stats and self.raw_data is not None:
            return float(self.raw_data.median())
        else:
            return self.quantile(0.5)
    
    def quantile(
        self,
        pos: Optional[Union[float, list, np.ndarray]] = None,
        by: Union[float, int, list, np.ndarray] = 1.0
    ) -> Union[float, list]:
        """
        Calculate approximate quantiles from the frequency distribution.
        
        Parameters
        ----------
        pos : float, list, or np.ndarray, optional
            Quantile position(s). If None, returns [0.25] (first quartile).
        by : float, int, list, or np.ndarray, default=1.0
            Divisor for quantile positions. Can be:
            - 1.0: positions in [0, 1] (default)
            - 100: positions in [0, 100] (percentiles)
            - 4: positions in [0, 4] (quartiles as 1, 2, 3)
            - Array: custom divisors for each position
        
        Returns
        -------
        float or list
            Quantile value(s).
        
        Examples
        --------
        >>> fdt.quantile(0.5)  # Median (by=1.0 default)
        >>> fdt.quantile(50, by=100)  # 50th percentile
        >>> fdt.quantile([1, 2, 3], by=4)  # Quartiles
        >>> fdt.quantile([0.1, 0.5, 0.9], by=[0.3, 0.5, 0.7])  # Custom
        """
        
        if pos is None:
            pos = [0.25]
        
        if self.use_raw_data_stats and self.raw_data is not None:
            if isinstance(pos, (int, float)):
                if isinstance(by, (int, float)):
                    adjusted_pos = float(pos) / float(by)
                else:
                    adjusted_pos = float(pos)
                return float(self.raw_data.quantile(adjusted_pos))
            else:
                result = []
                if isinstance(by, (int, float)):
                    for p in pos:
                        adjusted_pos = float(p) / float(by)
                        result.append(float(self.raw_data.quantile(adjusted_pos)))
                else:
                    if len(pos) != len(by):
                        raise ValueError("pos and by must have the same length when both are arrays")
                    for p, b in zip(pos, by):
                        adjusted_pos = float(p) / float(b)
                        result.append(float(self.raw_data.quantile(adjusted_pos)))
                return result
        
        def single_quantile(a: float, b: float = 1.0) -> float:
            if b <= 0:
                raise ValueError(f"by must be positive - got {b}")
            
            adjusted_pos = a / b
            
            if adjusted_pos < 0.0 or adjusted_pos > 1.0:
                raise ValueError(f"Quantile position should be between 0 and {b} after division - got {a}/{b} = {adjusted_pos}")
            
            pos_count = self.count * adjusted_pos
            
            cumulative_freq = self.table["cf"].values
            if pos_count > cumulative_freq[-1]:
                idx = len(cumulative_freq) - 1
            else:
                idx = np.where(pos_count <= cumulative_freq)[0][0]
            
            bins = self.binning.bins
            h = self.binning.h
            ll = bins[idx]
            cf_prev = 0 if idx < 1 else self.table.iloc[idx - 1, 4]
            f_q = self.table.iloc[idx, 1]
            
            if f_q == 0:
                return ll
            
            return ll + ((pos_count - cf_prev) * h) / f_q
        
        if isinstance(pos, (int, float)):
            if isinstance(by, (int, float)):
                return single_quantile(float(pos), float(by))
            else:
                return single_quantile(float(pos), 1.0)
        else:
            if isinstance(by, (int, float)):
                b = float(by)
                return [single_quantile(float(x), b) for x in pos]
            else:
                if len(pos) != len(by):
                    raise ValueError("pos and by must have the same length when both are arrays")
                return [single_quantile(float(p), float(b)) for p, b in zip(pos, by)]
    
    @lru_cache
    def var(self) -> float:
        if self.use_raw_data_stats and self.raw_data is not None:
            return float(self.raw_data.var(ddof=1))
        else:
            mean_val = self.mean()
            return np.sum((self._midpoints() - mean_val) ** 2 * self.table["f"]) / (self.count - 1)
    
    @lru_cache
    def sd(self) -> float:
        if self.use_raw_data_stats and self.raw_data is not None:
            return float(self.raw_data.std(ddof=1))
        else:
            return np.sqrt(self.var())
    
    @lru_cache
    def mfv(self) -> pd.Series:
        freqs = self.table["f"].to_numpy()
        bins = self.binning.bins
        h = self.binning.h
        
        def calculate_mfv(pos: int) -> float:
            lower_limit = bins[pos]
            current_freq = float(freqs[pos])
            preceding_freq = float(0 if pos - 1 < 0 else freqs[pos - 1])
            succeeding_freq = float(0 if pos + 1 >= len(freqs) else freqs[pos + 1])
            
            d1 = current_freq - preceding_freq
            d2 = current_freq - succeeding_freq
            
            if d1 + d2 == 0:
                return lower_limit + h / 2
            
            return float(lower_limit + (d1 / (d1 + d2)) * h)
        
        positions = np.where(freqs == freqs.max())[0]
        return pd.Series([calculate_mfv(pos) for pos in positions])
    
    def get_table(self) -> pd.DataFrame:
        return self.table
    
    def __repr__(self):
        table = self.table
        
        max_class_len = max([len(str(cls)) for cls in table['Class limits']] + [len('Class limits')])
        max_f_len = max([len(str(int(f))) for f in table['f']] + [1])
        max_rf_len = max([len(f"{rf:.3f}".rstrip('0').rstrip('.')) for rf in table['rf']] + [2])
        max_rfp_len = max([len(f"{rfp:.1f}") for rfp in table['rf(%)']] + [5])
        max_cf_len = max([len(str(int(cf))) for cf in table['cf']] + [2])
        max_cfp_len = max([len(f"{cfp:.1f}") for cfp in table['cf(%)']] + [5])
        
        class_width = max(max_class_len, 15)
        f_width = max(max_f_len, 4)
        rf_width = max(max_rf_len, 6)
        rfp_width = max(max_rfp_len, 6)
        cf_width = max(max_cf_len, 4)
        cfp_width = max(max_cfp_len, 6)
        
        header = (f"{'Class limits':{class_width}} "
                 f"{'f':>{f_width}} "
                 f"{'rf':>{rf_width}} "
                 f"{'rf(%)':>{rfp_width}} "
                 f"{'cf':>{cf_width}} "
                 f"{'cf(%)':>{cfp_width}}")
        
        total_width = (class_width + f_width + rf_width + rfp_width + 
                      cf_width + cfp_width + 5)
        
        result = header + "\n" + "-" * total_width + "\n"
        
        for i in range(len(table)):
            row = table.iloc[i]
            class_lim = str(row['Class limits'])[:class_width]
            f_val = int(row['f'])
            
            rf_str = f"{row['rf']:.3f}".rstrip('0').rstrip('.')
            if rf_str == '':
                rf_str = '0'
            
            rf_percent = f"{row['rf(%)']:.1f}"
            cf_val = int(row['cf'])
            cf_percent = f"{row['cf(%)']:.1f}"
            
            result += (f"{class_lim:{class_width}} "
                      f"{f_val:>{f_width}d} "
                      f"{rf_str:>{rf_width}} "
                      f"{rf_percent:>{rfp_width}} "
                      f"{cf_val:>{cf_width}d} "
                      f"{cf_percent:>{cfp_width}}\n")
        
        result += "-" * total_width
        return result
    
    def plot(
        self,
        type_: str = "fh",
        v: bool = False,
        v_round: int = 2,
        v_pos: int = 3,
        xlab: str = "Class limits",
        xlas: int = 0,
        ylab: Optional[str] = None,
        color: str = "gray",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        main: Optional[str] = None,
        edgecolor: str = "black",
        linewidth: int = 1,
        x_round: int = 2,
        show: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> None:
        
        """
        Create various types of plots for numerical frequency distribution.
        
        Parameters
        ----------
        type_ : str, default="fh"
            Plot type:
            - "fh": Frequency histogram (bar plot)
            - "fp": Frequency polygon (line plot)
            - "rfh": Relative frequency histogram
            - "rfp": Relative frequency polygon
            - "rfph": Relative frequency percentage histogram
            - "rfpp": Relative frequency percentage polygon
            - "d": Density plot
            - "cdh": Cumulative density histogram
            - "cdp": Cumulative density polygon
            - "cfh": Cumulative frequency histogram
            - "cfp": Cumulative frequency polygon
            - "cfph": Cumulative frequency percentage histogram
            - "cfpp": Cumulative frequency percentage polygon
        
        v : bool, default=False
            Whether to display frequency values on the plot.
        
        v_round : int, default=2
            Number of decimal places for value labels.
        
        v_pos : int, default=3
            Vertical position adjustment for value labels.
        
        xlab : str, default="Class limits"
            X-axis label.
        
        xlas : int, default=0
            Rotation of x-axis labels (0=horizontal, 1=vertical).
        
        ylab : str, optional
            Y-axis label (auto-generated based on plot type).
        
        color : str, default="gray"
            Bar/line color.
        
        xlim : tuple, optional
            X-axis limits.
        
        ylim : tuple, optional
            Y-axis limits.
        
        main : str, optional
            Plot title.
        
        edgecolor : str, default="black"
            Edge color for bars.
        
        linewidth : int, default=1
            Line width for polygon plots.
        
        x_round : int, default=2
            Number of decimal places for x-axis labels.
        
        show : bool, default=True
            Whether to display the plot.
        
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.
        
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plotting functions.
        """
        
        bins = self.binning.bins
        mids = self._midpoints()

        if xlim is None:
            xlim = (self.binning.start, self.binning.end)

        if main is None:
            main = self._column

        def make_range_labels(ax, ylim_):
            ybot, ytop = ylim_
            yrange = ytop - ybot
            yticks = np.arange(ybot, ybot + 1.1 * yrange, 0.1)
            ylabels = [f"{k*100:.0f}%" for k in yticks]
            ax.set_yticks(yticks, ylabels)

        def aux_barplot(
            ax,
            y,
            percent=False,
            default_ylab="Frequency",
        ) -> None:
            ylim_ = ylim or (0, y.max())
            
            if ax is None:
                fig, ax = plt.subplots()
                created_ax = True
            else:
                created_ax = False
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim_)
            if main is not None:
                ax.set_title(main)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab or default_ylab)
            ax.set_xticks(bins)

            if percent:
                make_range_labels(ax, ylim_)

            ax.bar(
                x=mids,
                height=y,
                width=self.binning.h,
                edgecolor=edgecolor,
                linewidth=linewidth,
                color=color,
                **kwargs,
            )

            if v:
                for xpos, ypos in zip(mids, y):
                    ax.text(
                        xpos,
                        ypos,
                        f"{ypos:.{v_round}f}",
                        va="bottom",
                        ha="center",
                        **kwargs,
                    )

            if show and created_ax:
                plt.show()

        def aux_polyplot(
            ax,
            y,
            percent=False,
            default_ylab="Frequency",
        ) -> None:
            ylim_ = ylim or (-0.1, y.max() + 0.1)
            
            if ax is None:
                fig, ax = plt.subplots()
                created_ax = True
            else:
                created_ax = False
                
            ax.set_xlim(xlim)
            ax.set_ylim(ylim_)
            if main is not None:
                ax.set_title(main)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab or default_ylab)
            ax.set_xticks(bins)

            if percent:
                make_range_labels(ax, ylim_)

            ax.plot(mids, y, "o-", color=color, **kwargs)

            if v:
                for xpos, ypos in zip(mids, y):
                    ax.text(
                        xpos,
                        ypos,
                        f"{ypos:.{v_round}f}",
                        va="bottom",
                        ha="center",
                        **kwargs,
                    )

            if show and created_ax:
                plt.show()

        if type_ == "fh":
            y = self.table["f"].to_numpy()
            aux_barplot(ax, y)
        elif type_ == "fp":
            y = self.table["f"].to_numpy()
            aux_polyplot(ax, y)
        elif type_ == "rfh":
            y = self.table["rf"].to_numpy()
            aux_barplot(ax, y)
        elif type_ == "rfp":
            y = self.table["rf"].to_numpy()
            aux_polyplot(ax, y)
        elif type_ == "rfph":
            y = self.table["rf"].to_numpy()
            aux_barplot(ax, y, percent=True)
        elif type_ == "rfpp":
            y = self.table["rf"].to_numpy()
            aux_polyplot(ax, y, percent=True)
        elif type_ == "d":
            y = self.table["rf"].to_numpy() / self.binning.h
            aux_barplot(ax, y, default_ylab="Density")
        elif type_ == "cdh":
            y = self.table["cf"].to_numpy() / (self.count * self.binning.h)
            aux_barplot(ax, y, default_ylab="Cumulative density")
        elif type_ == "cdp":
            y = self.table["cf"].to_numpy() / (self.count * self.binning.h)
            aux_polyplot(ax, y, default_ylab="Cumulative density")
        elif type_ == "cfh":
            y = self.table["cf"].to_numpy()
            aux_barplot(ax, y)
        elif type_ == "cfp":
            y = self.table["cf"].to_numpy()
            aux_polyplot(ax, y)
        elif type_ == "cfph":
            y = self.table["cf"].to_numpy() / self.count
            aux_barplot(ax, y, percent=True)
        elif type_ == "cfpp":
            y = self.table["cf"].to_numpy() / self.count
            aux_polyplot(ax, y, percent=True)
        else:
            raise ValueError(f"unknown plot type {repr(type_)}")
    
    @staticmethod
    def _make_table_from_data(data: pd.Series, binning: Binning, 
                             right: bool, round_: int) -> pd.DataFrame:
        categories = pd.cut(data, bins=binning.bins, right=right, include_lowest=True)
        freqs = categories.value_counts().sort_index()
        
        return NumericalFDT._make_table_from_frequencies(freqs, binning, right, round_)
    
    @staticmethod
    def _make_table_from_frequencies(freqs: pd.Series, binning: Binning,
                                    right: bool, round_: int) -> pd.DataFrame:
        classes = binning.format_classes(round_=round_, right=right)
        n = freqs.sum()
        rf = freqs / n
        rfp = rf * 100
        cf = freqs.cumsum()
        cfp = cf / n * 100
        
        table = pd.DataFrame({
            "Class limits": classes,
            "f": freqs.values,
            "rf": rf.values,
            "rf(%)": rfp.values,
            "cf": cf.values,
            "cf(%)": cfp.values,
        })
        
        return table
