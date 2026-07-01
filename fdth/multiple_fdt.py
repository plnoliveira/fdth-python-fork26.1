from typing import Optional, Any, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .utils import deduce_fdt_kind
from .numerical_fdt import NumericalFDT
from .categorical_fdt import CategoricalFDT

class MultipleFDT:
    """
    Container for multiple Frequency Distribution Tables.
    
    Attributes
    ----------
    fdts : dict
        Dictionary mapping column names to FDT objects.
    _data : pd.DataFrame
        Original input data.
    _kwargs : dict
        Additional keyword arguments passed to individual FDTs.
    _columns : pd.Index
        Column names from the input data.
    """
    
    def __init__(self, data: pd.DataFrame | np.ndarray, by: Optional[Union[str, list[str]]] = None, **kwargs) -> None:
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame({'V1': data})
            else:
                data = pd.DataFrame(data)
                if data.columns.dtype == 'int64':
                    data.columns = [f'V{i+1}' for i in range(data.shape[1])]
        
        self._by = by
        self._data = data
        self._kwargs = kwargs
        
        if by is None:
            if isinstance(data, pd.DataFrame):
                self._columns = data.columns        
            self._create_fdts()
            self._groups = None
        else:
            # Converter para lista se for string única
            if isinstance(by, str):
                by_cols = [by]
            else:
                by_cols = by
            
            self._by_cols = by_cols
            self._columns = [col for col in data.columns if col not in by_cols]
            self._create_grouped_fdts()
    
    def _create_fdts(self) -> None:
        self.fdts = {}
        
        for col_name, col_data in self._data.items():
            if deduce_fdt_kind(col_data) == "categorical":
                cat_kwargs = self._filter_kwargs_for_categorical(self._kwargs)
                self.fdts[col_name] = CategoricalFDT(col_data, **cat_kwargs)
            else:
                num_kwargs = self._filter_kwargs_for_numerical(self._kwargs)
                if self._kwargs.get('use_raw_data_stats', False) and not hasattr(col_data, 'iloc'):
                    num_kwargs['use_raw_data_stats'] = False
                self.fdts[col_name] = NumericalFDT(col_data, **num_kwargs)
    
    def _create_grouped_fdts(self):
        # Verificar se todas as colunas de agrupamento existem
        for col in self._by_cols:
            if col not in self._data.columns:
                raise ValueError(f"Grouping column '{col}' not found in DataFrame")
            
            if deduce_fdt_kind(self._data[col]) != "categorical":
                raise ValueError(f"Grouping column '{col}' must be categorical")
        
        # Criar chave composta para agrupamento
        group_cols = [self._data[col] for col in self._by_cols]
        if len(group_cols) == 1:
            group_series = group_cols[0]
        else:
            # Combinar múltiplas colunas em uma única chave
            group_series = pd.Series(
                [tuple(row) for row in zip(*group_cols)],
                index=self._data.index
            )
        
        data_to_analyze = self._data.drop(columns=self._by_cols)
        unique_groups = group_series.unique()
        
        self._groups = {}
        for group in unique_groups:
            group_data = data_to_analyze[group_series == group]
            group_fdts = {}
            
            for col_name, col_data in group_data.items():
                if deduce_fdt_kind(col_data) == "categorical":
                    cat_kwargs = self._filter_kwargs_for_categorical(self._kwargs)
                    group_fdts[col_name] = CategoricalFDT(col_data, **cat_kwargs)
                else:
                    num_kwargs = self._filter_kwargs_for_numerical(self._kwargs)
                    if self._kwargs.get('use_raw_data_stats', False) and not hasattr(col_data, 'iloc'):
                        num_kwargs['use_raw_data_stats'] = False
                    group_fdts[col_name] = NumericalFDT(col_data, **num_kwargs)
            
            # Converter grupo para string se for tupla
            if isinstance(group, tuple):
                group_key = ', '.join(str(g) for g in group)
            else:
                group_key = group
                
            self._groups[group_key] = group_fdts
        
        self.fdts = {}
    
    def _filter_kwargs_for_numerical(self, kwargs: dict) -> dict:
        filtered = kwargs.copy()
        categorical_params = ['sort', 'decreasing']
        for param in categorical_params:
            filtered.pop(param, None)
        return filtered
    
    def _filter_kwargs_for_categorical(self, kwargs: dict) -> dict:
        filtered = kwargs.copy()
        numerical_params = [
            'binning', 'start', 'end', 'h', 'k', 'right', 
            'na_rm', 'round_', 'use_raw_data_stats'
        ]
        for param in numerical_params:
            filtered.pop(param, None)
        return filtered
        
    def get_group(self, group: Union[str, tuple]):
        """
        Get all FDT objects for a specific group.
        
        Parameters
        ----------
        group : str or tuple
            Group name or tuple of group values.
        
        Returns
        -------
        dict
            Dictionary mapping variable names to FDT objects.
        """
        if self._by is None:
            return None
        
        # Converter para string se for tupla
        if isinstance(group, tuple):
            group_key = '_'.join(str(g) for g in group)
        else:
            group_key = group
        
        return self._groups.get(group_key)

    def get_groups(self):
        """Return list of all group keys."""
        return list(self._groups.keys()) if self._by is not None else []
    
    def print_group(self, group: Union[str, tuple]):
        """
        Print all FDTs for a specific group in formatted way.
        
        Parameters
        ----------
        group : str or tuple
            Group name or tuple of group values.
        """
        if self._by is None:
            print("No groups available (not using 'by' parameter)")
            return
        
        if isinstance(group, tuple):
            group_key = '_'.join(str(g) for g in group)
        else:
            group_key = group
        
        if group_key not in self._groups:
            print(f"Group '{group_key}' not found")
            return
        
        # Mostrar os valores originais do grupo se for tupla
        if isinstance(group, tuple):
            group_display = ' x '.join(str(g) for g in group)
        else:
            group_display = group
        
        print(f"\n{'='*60}")
        print(f"GROUP: {group_display}")
        print('='*60)
        
        grupo_fdts = self._groups[group_key]
        for var_name, fdt_obj in grupo_fdts.items():
            print(f"\n{var_name}")
            print("-" * 40)
            print(fdt_obj)
            print()
    
    def __getitem__(self, key):
        if self._by is not None:
            if isinstance(key, tuple):
                if len(key) == 2:
                    # (grupo, coluna)
                    group, column = key
                    if isinstance(group, tuple):
                        group_key = '_'.join(str(g) for g in group)
                    else:
                        group_key = group
                    return self._groups[group_key][column]
                else:
                    # Múltiplas dimensões de grupo
                    group_vals = key[:-1]
                    column = key[-1]
                    group_key = '_'.join(str(g) for g in group_vals)
                    return self._groups[group_key][column]
            elif key in self._columns:
                result = {}
                for group_key, group_fdts in self._groups.items():
                    if key in group_fdts:
                        result[group_key] = group_fdts[key]
                return result
            else:
                raise KeyError(f"Key '{key}' not found")
        else:
            return self.fdts[key]
    
    def __contains__(self, key):
        if self._by is not None:
            if isinstance(key, tuple):
                if len(key) == 2:
                    group, column = key
                    if isinstance(group, tuple):
                        group_key = '_'.join(str(g) for g in group)
                    else:
                        group_key = group
                    return group_key in self._groups and column in self._groups[group_key]
                else:
                    group_vals = key[:-1]
                    column = key[-1]
                    group_key = '_'.join(str(g) for g in group_vals)
                    return group_key in self._groups and column in self._groups[group_key]
            else:
                return any(key in group_fdts for group_fdts in self._groups.values())
        else:
            return key in self.fdts
    
    def keys(self):
        if self._by is not None:
            return list(self._columns)
        else:
            return list(self.fdts.keys())
    
    def values(self):
        if self._by is not None:
            result = {}
            for col in self._columns:
                result[col] = self[col]
            return result.values()
        else:
            return self.fdts.values()
    
    def items(self):
        if self._by is not None:
            result = {}
            for col in self._columns:
                result[col] = self[col]
            return result.items()
        else:
            return self.fdts.items()
    
    def mean(self):
        if self._by is not None:
            result = pd.DataFrame()
            for group, group_fdts in self._groups.items():
                for col_name, fdt in group_fdts.items():
                    if isinstance(fdt, NumericalFDT):
                        result.at[group, col_name] = fdt.mean()
            return result
        else:
            result = {}
            for col_name, fdt in self.fdts.items():
                if isinstance(fdt, NumericalFDT):
                    result[col_name] = fdt.mean()
            
            output_lines = []
            for col_name, value in result.items():
                output_lines.append(f"{col_name:<15} {value:.4f}")
            return "\n".join(output_lines)
    
    def median(self, by: Union[float, int] = 1.0):
        if self._by is not None:
            result = pd.DataFrame()
            for group, group_fdts in self._groups.items():
                for col_name, fdt_obj in group_fdts.items():
                    if hasattr(fdt_obj, 'median'):
                        try:
                            result.at[group, col_name] = fdt_obj.median(by=by)
                        except (TypeError, AttributeError, KeyError):
                            pass
            return result
        else:
            result = {}
            for col_name, fdt_obj in self.fdts.items():
                if hasattr(fdt_obj, 'median'):
                    try:
                        result[col_name] = fdt_obj.median(by=by)
                    except (TypeError, AttributeError, KeyError):
                        pass
            
            output_lines = []
            for col_name, value in result.items():
                if isinstance(value, (int, float, np.number)):
                    output_lines.append(f"{col_name:<15} {value:.4f}")
                else:
                    output_lines.append(f"{col_name:<15} {value}")
            return "\n".join(output_lines)
    
    def var(self):
        if self._by is not None:
            result = pd.DataFrame()
            for group, group_fdts in self._groups.items():
                for col_name, fdt in group_fdts.items():
                    if isinstance(fdt, NumericalFDT):
                        result.at[group, col_name] = fdt.var()
            return result
        else:
            result = {}
            for col_name, fdt in self.fdts.items():
                if isinstance(fdt, NumericalFDT):
                    result[col_name] = fdt.var()
            
            output_lines = []
            for col_name, value in result.items():
                output_lines.append(f"{col_name:<15} {value:.4f}")
            return "\n".join(output_lines)
    
    def sd(self):
        if self._by is not None:
            result = pd.DataFrame()
            for group, group_fdts in self._groups.items():
                for col_name, fdt in group_fdts.items():
                    if isinstance(fdt, NumericalFDT):
                        result.at[group, col_name] = fdt.sd()
            return result
        else:
            result = {}
            for col_name, fdt in self.fdts.items():
                if isinstance(fdt, NumericalFDT):
                    result[col_name] = fdt.sd()
            
            output_lines = []
            for col_name, value in result.items():
                output_lines.append(f"{col_name:<15} {value:.4f}")
            return "\n".join(output_lines)
    
    def quantile(self, probs: Union[float, list] = None, by: Union[float, int, list, np.ndarray] = 1.0):
        if probs is None:
            probs = [0.25]
        
        if self._by is not None:
            result = {}
            for group, group_fdts in self._groups.items():
                group_result = {}
                for col_name, fdt in group_fdts.items():
                    if hasattr(fdt, 'quantile'):
                        try:
                            group_result[col_name] = fdt.quantile(probs, by=by)
                        except (TypeError, AttributeError):
                            pass
                result[group] = group_result
            
            if isinstance(probs, (int, float)):
                df_result = pd.DataFrame()
                for group, group_result in result.items():
                    for col_name, value in group_result.items():
                        df_result.at[group, col_name] = value
                return df_result
            else:
                multi_index = pd.MultiIndex.from_product([result.keys(), result[list(result.keys())[0]].keys()])
                df_result = pd.DataFrame(index=multi_index, columns=[f'Q{i+1}' for i in range(len(probs))])
                for group, group_result in result.items():
                    for col_name, values in group_result.items():
                        if isinstance(values, list):
                            for i, val in enumerate(values):
                                df_result.at[(group, col_name), f'Q{i+1}'] = val
                return df_result
        else:
            result = {}
            for col_name, fdt in self.fdts.items():
                if hasattr(fdt, 'quantile'):
                    try:
                        result[col_name] = fdt.quantile(probs, by=by)
                    except (TypeError, AttributeError):
                        pass
            
            if isinstance(probs, (int, float)):
                return pd.Series(result)
            else:
                df_result = pd.DataFrame(index=result.keys(), columns=[f'Q{i+1}' for i in range(len(probs))])
                for col_name, values in result.items():
                    if isinstance(values, list):
                        for i, val in enumerate(values):
                            df_result.at[col_name, f'Q{i+1}'] = val
                return df_result
    
    def mfv(self, round_decimals: int = 2):
        """
        Return modal values as a DataFrame.
        
        Parameters
        ----------
        round_decimals : int, default=2
            Number of decimal places to round numerical MFV values.
            Set to None to keep original precision.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with modal values.
        """
        if self._by is not None:
            all_modes = []
            for group, group_fdts in self._groups.items():
                for col_name, fdt_obj in group_fdts.items():
                    modes = fdt_obj.mfv()
                    if hasattr(modes, 'tolist'):
                        modes_list = modes.tolist()
                    else:
                        modes_list = list(modes) if isinstance(modes, pd.Series) else [modes]
                    
                    for i, mode_val in enumerate(modes_list):
                        all_modes.append({
                            'Group': group,
                            'Variable': col_name,
                            'MFV': mode_val,
                            'Index': i
                        })
            
            df = pd.DataFrame(all_modes) if all_modes else pd.DataFrame()
            
            if round_decimals is not None:
                df_display = df.copy()
                if 'MFV' in df_display.columns:
                    def format_value(x):
                        if isinstance(x, (int, float, np.number)):
                            return round(float(x), round_decimals)
                        return x
                    df_display['MFV'] = df_display['MFV'].apply(format_value)
                return df_display
            return df
        else:
            all_modes = []
            for col_name, fdt_obj in self.fdts.items():
                modes = fdt_obj.mfv()
                if hasattr(modes, 'tolist'):
                    modes_list = modes.tolist()
                else:
                    modes_list = list(modes) if isinstance(modes, pd.Series) else [modes]
                
                for i, mode_val in enumerate(modes_list):
                    all_modes.append({
                        'Variable': col_name,
                        'MFV': mode_val,
                        'Index': i
                    })
            
            df = pd.DataFrame(all_modes) if all_modes else pd.DataFrame()
            
            if round_decimals is not None:
                df_display = df.copy()
                if 'MFV' in df_display.columns:
                    def format_value(x):
                        if isinstance(x, (int, float, np.number)):
                            return round(float(x), round_decimals)
                        return x
                    df_display['MFV'] = df_display['MFV'].apply(format_value)
                return df_display
            return df
    
    def plot(self, numeric_type: str = "fh", categorical_type: str = "fb", 
             grouped: bool = True, **kwargs):
        
        """
        Create plots for multiple frequency distribution tables.
        
        Parameters
        ----------
        numeric_type : str, default="fh"
            Plot type for numerical variables.
            See NumericalFDT.plot() for available types.
        
        categorical_type : str, default="fb"
            Plot type for categorical variables.
            See CategoricalFDT.plot() for available types.
        
        grouped : bool, default=True
            For grouped data (when 'by' parameter is used):
            - True: All groups and variables in a single multi-plot grid
            - False: Separate multi-plot grid for each group
        
        **kwargs : dict
            Additional keyword arguments passed to individual FDT plot methods.
        
        Notes
        -----
        Automatically paginates when there are more than 12 plots.
        Creates subplot grids with up to 4 columns per page.
        """
        
        if self._by is not None:
            groups = list(self._groups.keys())
            
            if grouped:
                all_fdts = []
                for group in groups:
                    group_fdts = self._groups[group]
                    for col_name, fdt in group_fdts.items():
                        all_fdts.append((f"{group} - {col_name}", fdt))
                
                n_total = len(all_fdts)
                max_per_page = 12
                n_pages = (n_total + max_per_page - 1) // max_per_page
                
                for page in range(n_pages):
                    start_idx = page * max_per_page
                    end_idx = min((page + 1) * max_per_page, n_total)
                    page_fdts = all_fdts[start_idx:end_idx]
                    page_total = len(page_fdts)
                    
                    if n_pages > 1:
                        print(f"\nPage {page + 1} of {n_pages}")
                    
                    n_cols = min(4, page_total)
                    n_rows = (page_total + n_cols - 1) // n_cols
                    
                    fig_width = min(20, 5 * n_cols)
                    fig_height = min(15, 3.5 * n_rows)
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
                    
                    if n_rows == 1 and n_cols == 1:
                        axes = np.array([axes])
                    elif n_rows == 1:
                        axes = axes.flatten()
                    elif n_cols == 1:
                        axes = axes.flatten()
                    else:
                        axes = axes.flatten()
                    
                    for i, (title, fdt_obj) in enumerate(page_fdts):
                        if i >= len(axes):
                            break
                        
                        ax = axes[i]
                        if isinstance(fdt_obj, NumericalFDT):
                            fdt_obj.plot(type_=numeric_type, show=False, ax=ax, main=title, 
                                       v=False, **kwargs)
                        else:
                            fdt_obj.plot(type_=categorical_type, show=False, ax=ax, main=title,
                                       v=False, **kwargs)
                    
                    for i in range(len(page_fdts), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout(pad=2.5, h_pad=2.0, w_pad=1.5)
                    plt.show()
            else:
                for group in groups:
                    group_fdts = self._groups[group]
                    n_total = len(group_fdts)
                    max_per_page = 12
                    n_pages = (n_total + max_per_page - 1) // max_per_page
                    
                    for page in range(n_pages):
                        start_idx = page * max_per_page
                        end_idx = min((page + 1) * max_per_page, n_total)
                        page_items = list(group_fdts.items())[start_idx:end_idx]
                        page_total = len(page_items)
                        
                        if n_pages > 1:
                            print(f"\nGroup '{group}' - Page {page + 1} of {n_pages}")
                        
                        n_cols = min(4, page_total)
                        n_rows = (page_total + n_cols - 1) // n_cols
                        
                        fig_width = min(20, 5 * n_cols)
                        fig_height = min(15, 3.5 * n_rows)
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
                        
                        if n_rows == 1 and n_cols == 1:
                            axes = np.array([axes])
                        elif n_rows == 1:
                            axes = axes.flatten()
                        elif n_cols == 1:
                            axes = axes.flatten()
                        else:
                            axes = axes.flatten()
                        
                        for i, (col_name, fdt) in enumerate(page_items):
                            if i >= len(axes):
                                break
                            
                            ax = axes[i]
                            if isinstance(fdt, NumericalFDT):
                                fdt.plot(type_=numeric_type, show=False, ax=ax, 
                                       main=f"{group} - {col_name}", v=False, **kwargs)
                            else:
                                fdt.plot(type_=categorical_type, show=False, ax=ax,
                                       main=f"{group} - {col_name}", v=False, **kwargs)
                        
                        for i in range(len(page_items), len(axes)):
                            axes[i].set_visible(False)
                        
                        plt.tight_layout(pad=2.5, h_pad=2.0, w_pad=1.5)
                        plt.show()
        else:
            n_total = len(self.fdts)
            max_per_page = 12
            n_pages = (n_total + max_per_page - 1) // max_per_page
            items = list(self.fdts.items())
            
            for page in range(n_pages):
                start_idx = page * max_per_page
                end_idx = min((page + 1) * max_per_page, n_total)
                page_items = items[start_idx:end_idx]
                page_total = len(page_items)
                
                if n_pages > 1:
                    print(f"\nPage {page + 1} of {n_pages}")
                
                n_cols = min(4, page_total)
                n_rows = (page_total + n_cols - 1) // n_cols
                
                fig_width = min(20, 5 * n_cols)
                fig_height = min(15, 3.5 * n_rows)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
                
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                elif n_rows == 1:
                    axes = axes.flatten()
                elif n_cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                
                for i, (col_name, fdt_obj) in enumerate(page_items):
                    if i >= len(axes):
                        break
                    
                    ax = axes[i]
                    if isinstance(fdt_obj, NumericalFDT):
                        fdt_obj.plot(type_=numeric_type, show=False, ax=ax, 
                                   main=col_name, v=False, **kwargs)
                    else:
                        fdt_obj.plot(type_=categorical_type, show=False, ax=ax,
                                   main=col_name, v=False, **kwargs)
                
                for i in range(len(page_items), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout(pad=2.5, h_pad=2.0, w_pad=1.5)
                plt.show()
    
    def __repr__(self):
        if self._by is not None:
            result = ""
            
            for group_key, group_fdts in self._groups.items():
                # Tentar mostrar grupo de forma legível
                if '_' in group_key and len(self._by_cols) > 1:
                    group_vals = group_key.split('_')
                    if len(group_vals) == len(self._by_cols):
                        group_display = ' x '.join(group_vals)
                    else:
                        group_display = group_key
                else:
                    group_display = group_key
                
                result += f"\n{'='*50}\n"
                result += f"{group_display}\n"
                result += f"{'='*50}\n"
                
                for col_name, fdt in group_fdts.items():
                    result += f"\n{col_name}\n"
                    result += repr(fdt) + "\n"
            
            return result
        else:
            result = "MultipleFDT\n"
            for col_name, fdt_obj in self.fdts.items():
                result += f"\n{col_name}\n"
                result += repr(fdt_obj) + "\n"
            return result
    
    def __str__(self):
        return self.__repr__()
