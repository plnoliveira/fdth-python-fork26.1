from typing import Optional, Any, Union
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CategoricalFDT:
    """
    Frequency Distribution Table for categorical data.
    
    Attributes
    ----------
    count : int
        Total number of observations.
    table : pd.DataFrame
        Frequency distribution table with columns:
        - Category: Category labels
        - f: Absolute frequency
        - rf: Relative frequency
        - rf(%): Relative frequency percentage
        - cf: Cumulative frequency
        - cf(%): Cumulative frequency percentage
    """
    
    def __init__(
        self,
        data: Optional[pd.Series | list] = None,
        *,
        freqs: Optional[pd.Series | dict[Any, int]] = None,
        sort: bool = True,
        decreasing: bool = False,
    ) -> None:
        if data is not None:
            if freqs is not None:
                raise ValueError("`data` and `freqs` must not be both specified")

            self._column = data.name
            data = pd.Series(data).astype("category")

            self.count, self.table = self._make_table_from_data(
                data, sort=sort, decreasing=decreasing
            )
        elif freqs is not None:
            if data is not None:
                raise ValueError("`data` and `freqs` must not be both specified")

            if isinstance(freqs, dict):
                freqs = pd.Series(freqs)
            elif isinstance(freqs, pd.Series):
                freqs = freqs
            else:
                raise TypeError("`freqs` must be dict | pandas.Series")

            self.count, self.table = self._make_table_from_frequencies(
                freqs, sort=sort, decreasing=decreasing
            )
        else:
            raise ValueError("one of `data` or `table` must be specified")

    def get_table(self) -> pd.DataFrame:
        return self.table

    @lru_cache
    def mean(self) -> float:
        raise TypeError("Categorical data does not have a mean")

    @lru_cache
    def median(self, by: Union[float, int] = 1.0) -> Any:
        """
        Calculate the median category.
        
        Parameters
        ----------
        by : float or int, default=1.0
            Divisor for median position.
        """
        if len(self.table) == 0:
            return None
        
        median_pos = 0.5 / float(by)
        pos_count = self.count * median_pos
        
        for i, row in self.table.iterrows():
            if row['cf'] >= pos_count:
                return row['Category']
        return self.table.iloc[-1]['Category']

    @lru_cache
    def var(self) -> float:
        raise TypeError("Categorical data does not have variance")

    @lru_cache
    def sd(self) -> float:
        raise TypeError("Categorical data does not have standard deviation")

    def quantile(self, pos: Union[float, list] = None, by: Union[float, int] = 1.0) -> Union[Any, list]:
        """
        Calculate quantiles for categorical data.
        
        Parameters
        ----------
        pos : float or list
            Quantile position(s) between 0 and 1 (or between 0 and `by`).
        by : float or int, default=1.0
            Divisor for quantile positions.
            - 1.0: positions in [0, 1] (default)
            - 100: positions in [0, 100] (percentiles)
            - 4: positions in [0, 4] (quartiles)
        
        Returns
        -------
        Any or list
            Quantile category value(s).
        """
        if pos is None:
            pos = [0.25]
        
        if isinstance(pos, (int, float)):
            pos = [pos]
        
        result = []
        
        for p in pos:

            adjusted_p = float(p) / float(by)
            
            if adjusted_p < 0 or adjusted_p > 1:
                raise ValueError(f"Quantile position should be between 0 and {by} after division - got {p}/{by} = {adjusted_p}")
            
            pos_count = self.count * adjusted_p
            
            found = False
            for i, row in self.table.iterrows():
                if row['cf'] >= pos_count:
                    result.append(row['Category'])
                    found = True
                    break
            
            if not found:
                result.append(self.table.iloc[-1]['Category'])
        
        return result[0] if len(result) == 1 else result

    @lru_cache
    def mfv(self) -> pd.Series:
        freqs = self.table["f"].to_numpy()
        positions = np.where(freqs == freqs.max())[0]
        return pd.Series(self.table["Category"][i] for i in positions)

    @staticmethod
    def _make_table_from_frequencies(
        freqs: pd.Series | dict[Any, int], sort: bool, decreasing: bool
    ) -> tuple[int, pd.DataFrame]:
        if isinstance(freqs, dict):
            freqs = pd.Series(freqs)

        if sort:
            freqs = freqs.sort_values(ascending=not decreasing)

        count = freqs.sum()

        rf = freqs / count
        rfp = rf * 100
        cf = freqs.cumsum()
        cfp = rfp.cumsum()

        return count, pd.DataFrame({
            "Category": freqs.index,
            "f": freqs.values,
            "rf": rf.values,
            "rf(%)": rfp.values,
            "cf": cf.values,
            "cf(%)": cfp.values,
        })

    @staticmethod
    def _make_table_from_data(
        data: pd.Series, sort: bool, decreasing: bool
    ) -> tuple[int, pd.DataFrame]:
        data = data.astype("category")

        if len(data.cat.categories) == 0:
            raise ValueError("No valid categories found in the data.")

        freqs = data.value_counts(sort=False)

        return CategoricalFDT._make_table_from_frequencies(
            freqs=freqs, sort=sort, decreasing=decreasing
        )

    def __repr__(self):
        table = self.table
        
        max_category_len = max([len(str(cat)) for cat in table['Category']] + [len('Category')])
        max_f_len = max([len(str(int(f))) for f in table['f']] + [1])
        max_rf_len = max([len(f"{rf:.3f}".rstrip('0').rstrip('.')) for rf in table['rf']] + [2])
        max_rfp_len = max([len(f"{rfp:.1f}") for rfp in table['rf(%)']] + [5])
        max_cf_len = max([len(str(int(cf))) for cf in table['cf']] + [2])
        max_cfp_len = max([len(f"{cfp:.1f}") for cfp in table['cf(%)']] + [5])
        
        category_width = max(max_category_len, 15)
        f_width = max(max_f_len, 4)
        rf_width = max(max_rf_len, 6)
        rfp_width = max(max_rfp_len, 6)
        cf_width = max(max_cf_len, 4)
        cfp_width = max(max_cfp_len, 6)
        
        header = (f"{'Category':{category_width}} "
                 f"{'f':>{f_width}} "
                 f"{'rf':>{rf_width}} "
                 f"{'rf(%)':>{rfp_width}} "
                 f"{'cf':>{cf_width}} "
                 f"{'cf(%)':>{cfp_width}}")
        
        total_width = (category_width + f_width + rf_width + rfp_width + 
                      cf_width + cfp_width + 5)
        
        result = header + "\n" + "-" * total_width + "\n"
        
        for i in range(len(table)):
            row = table.iloc[i]
            category = str(row['Category'])[:category_width]
            f_val = int(row['f'])
            
            rf_str = f"{row['rf']:.3f}".rstrip('0').rstrip('.')
            if rf_str == '':
                rf_str = '0'
            
            rf_percent = f"{row['rf(%)']:.1f}"
            cf_val = int(row['cf'])
            cf_percent = f"{row['cf(%)']:.1f}"
            
            result += (f"{category:{category_width}} "
                      f"{f_val:>{f_width}d} "
                      f"{rf_str:>{rf_width}} "
                      f"{rf_percent:>{rfp_width}} "
                      f"{cf_val:>{cf_width}d} "
                      f"{cf_percent:>{cfp_width}}\n")
        
        result += "-" * total_width
        return result


    def plot(
        self,
        type_: str = "fb",
        v: bool = False,
        v_round: int = 2,
        v_pos: int = 3,
        xlab: Optional[str] = None,
        xlas: int = 0,
        ylab: Optional[str] = None,
        y2lab: Optional[str] = None,
        y2cfp=np.arange(0, 101, 25),
        col: str = "0.4",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        main: Optional[str] = None,
        edgecolor: str = "black",
        box: bool = False,
        show: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> None:
        
        """
        Create various types of plots for categorical frequency distribution.
        
        Parameters
        ----------
        type_ : str, default="fb"
            Plot type:
            - "fb": Frequency bar plot
            - "fp": Frequency polygon plot
            - "fd": Frequency dot chart
            - "pa": Pareto chart
            - "rfb": Relative frequency bar plot
            - "rfp": Relative frequency polygon plot
            - "rfd": Relative frequency dot chart
            - "rfpb": Relative frequency percentage bar plot
            - "rfpp": Relative frequency percentage polygon plot
            - "rfpd": Relative frequency percentage dot chart
            - "cfb": Cumulative frequency bar plot
            - "cfp": Cumulative frequency polygon plot
            - "cfd": Cumulative frequency dot chart
            - "cfpb": Cumulative frequency percentage bar plot
            - "cfpp": Cumulative frequency percentage polygon plot
            - "cfpd": Cumulative frequency percentage dot chart
        
        v : bool, default=False
            Whether to display values on the plot.
        
        v_round : int, default=2
            Decimal places for value labels.
        
        v_pos : int, default=3
            Position adjustment for value labels.
        
        xlab : str, optional
            X-axis label (auto-generated based on plot type).
        
        xlas : int, default=0
            Rotation of x-axis labels (0=horizontal, 1=vertical).
        
        ylab : str, optional
            Y-axis label (auto-generated based on plot type).
        
        y2lab : str, optional
            Secondary y-axis label for Pareto chart.
        
        y2cfp : array-like, default=np.arange(0, 101, 25)
            Tick positions for secondary y-axis in Pareto chart.
        
        col : str, default="0.4"
            Color for bars/lines.
        
        xlim : tuple, optional
            X-axis limits.
        
        ylim : tuple, optional
            Y-axis limits.
        
        main : str, optional
            Plot title.
        
        edgecolor : str, default="black"
            Edge color for bars.
        
        box : bool, default=False
            Whether to show complete box around plot.
        
        show : bool, default=True
            Whether to display the plot.
        
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.
        """    
        
        x = self.table

        if main is None:
            main = self._column

        def plot_b(ax, y, categories):
            bar_positions = np.arange(len(categories))

            ax.bar(bar_positions, y, color=col, edgecolor=edgecolor)
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(categories, rotation=xlas * 90)

            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab)
            if main:
                ax.set_title(main)
            if box:
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)

            if v:
                for i, val in enumerate(y):
                    ax.text(i, val, f"{round(val, v_round)}", ha="center", va="bottom")

        def plot_p(ax, y, categories):
            ax.plot(range(len(categories)), y, "o-", color=col, markersize=5)

            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=xlas * 90)

            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab)
            if main:
                ax.set_title(main)

            if v:
                for i, val in enumerate(y):
                    ax.text(i, val, f"{round(val, v_round)}", ha="center", va="bottom")

        def plot_d(ax, y, categories):
            ax.plot(y, range(len(categories)), "o", color=col)
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels(categories)

            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab)
            if main:
                ax.set_title(main)

            if v:
                for i, val in enumerate(y):
                    ax.text(val, i, f"{round(val, v_round)}", ha="right")

        def plot_pa(ax, y, cf, cfp, categories):
            bar_positions = np.arange(len(categories))

            ax.bar(bar_positions, y, color=col, edgecolor=edgecolor)
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(categories, rotation=xlas * 90)

            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab)
            if main:
                ax.set_title(main)

            ax.set_ylim(0, max(cf) * 1.1)

            ax2 = ax.twinx()
            ax2.plot(
                bar_positions, cf, color="blue", marker="o", linestyle="-", markersize=5
            )
            ax2.set_ylabel(y2lab or "Cumulative frequency, (%)")
            ax2.set_ylim(0, max(cf) * 1.1)

        created_ax = False
        if ax is None:
            fig, ax = plt.subplots()
            created_ax = True

        categories = x["Category"]
        if type_ == "fb":
            y = x.iloc[:, 1]
            xlab = xlab or "Category"
            ylab = ylab or "Frequency"
            if ylim is None:
                ax.set_ylim(0, max(y) * 1.3)
            plot_b(ax, y, categories)

        elif type_ == "fp":
            y = x.iloc[:, 1]
            xlab = xlab or "Category"
            ylab = ylab or "Frequency"
            if ylim is None:
                ax.set_ylim(0, max(y) * 1.2)
            plot_p(ax, y, categories)

        elif type_ == "fd":
            y = x.iloc[:, 1]
            xlab = xlab or "Frequency"
            plot_d(ax, y, categories)

        elif type_ == "pa":
            y = x.iloc[:, 1]
            cf = x.iloc[:, 4]
            cfp = x.iloc[:, 5]
            xlab = xlab or "Category"
            ylab = ylab or "Frequency"
            y2lab = y2lab or "Cumulative frequency, (%)"
            if ylim is None:
                ax.set_ylim(0, sum(y) * 1.1)
            plot_pa(ax, y, cf, cfp, categories)

        elif type_ == "rfb":
            y = x.iloc[:, 2]
            xlab = xlab or "Category"
            ylab = ylab or "Relative frequency"
            plot_b(ax, y, categories)

        elif type_ == "rfp":
            y = x.iloc[:, 2]
            xlab = xlab or "Category"
            ylab = ylab or "Relative frequency"
            if ylim is None:
                ax.set_ylim(0, max(y) * 1.2)
            plot_p(ax, y, categories)

        elif type_ == "rfd":
            y = x.iloc[:, 2]
            xlab = xlab or "Relative frequency"
            plot_d(ax, y, categories)

        elif type_ == "rfpb":
            y = x.iloc[:, 3]
            xlab = xlab or "Category"
            ylab = ylab or "Relative frequency (%)"
            plot_b(ax, y, categories)

        elif type_ == "rfpp":
            y = x.iloc[:, 3]
            xlab = xlab or "Category"
            ylab = ylab or "Relative frequency (%)"
            if ylim is None:
                ax.set_ylim(0, max(y) * 1.2)
            plot_p(ax, y, categories)

        elif type_ == "rfpd":
            y = x.iloc[:, 3]
            xlab = xlab or "Relative frequency (%)"
            plot_d(ax, y, categories)

        elif type_ == "cfb":
            y = x.iloc[:, 4]
            xlab = xlab or "Category"
            ylab = ylab or "Cumulative frequency"
            plot_b(ax, y, categories)

        elif type_ == "cfp":
            y = x.iloc[:, 4]
            xlab = xlab or "Category"
            ylab = ylab or "Cumulative frequency"
            if ylim is None:
                ax.set_ylim(0, max(y) * 1.2)
            plot_p(ax, y, categories)

        elif type_ == "cfd":
            y = x.iloc[:, 4]
            xlab = xlab or "Cumulative frequency"
            plot_d(ax, y, categories)

        elif type_ == "cfpb":
            y = x.iloc[:, 5]
            xlab = xlab or "Category"
            ylab = ylab or "Cumulative frequency (%)"
            plot_b(ax, y, categories)

        elif type_ == "cfpp":
            y = x.iloc[:, 5]
            xlab = xlab or "Category"
            ylab = ylab or "Cumulative frequency (%)"
            if ylim is None:
                ax.set_ylim(0, max(y) * 1.2)
            plot_p(ax, y, categories)

        elif type_ == "cfpd":
            y = x.iloc[:, 5]
            xlab = xlab or "Cumulative frequency (%)"
            plot_d(ax, y, categories)

        if show and created_ax:
            plt.tight_layout()
            plt.show()
