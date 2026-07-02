import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fdth import fdt, CategoricalFDT


class Test(unittest.TestCase):
    def test_categorical_fdt(self):
        data = ["Blue", "Red", "Blue", "Green", "Blue", "Red"]
        # CategoricalFDT expects a pandas Series (it reads `data.name`);
        # use `fdt(data)` if you want to pass a plain list.
        fd = CategoricalFDT(pd.Series(data))
        assert isinstance(fd, CategoricalFDT)

        # tabela
        assert isinstance(fd.table, pd.DataFrame)
        assert not fd.table.empty

        # testa o __repr__ (a representação é a tabela formatada,
        # cujo cabeçalho começa com a coluna "Category")
        repr_output = repr(fd)
        assert isinstance(repr_output, str)
        assert "Category" in repr_output

    def test_data_and_freqs(self):
        data = ["Blue", "Red", "Blue", "Green", "Blue", "Red"]
        fd_1 = CategoricalFDT(pd.Series(data))
        fd_2 = CategoricalFDT(freqs={"Blue": 3, "Red": 2, "Green": 1})
        fd_3 = fdt(freqs={"Blue": 3, "Red": 2, "Green": 1})
        # there is no `to_string()`; the string form is the __repr__ table
        self.assertEqual(str(fd_1), str(fd_2))
        self.assertEqual(str(fd_1), str(fd_3))

    def test_plot(self):
        data = ["Blue", "Red", "Blue", "Green", "Blue", "Red"]
        fd = fdt(data)
        types = ["fb", "fp", "fd", "pa", "rfb", "rfp", "rfd", "rfpb", "rfpp", "rfpd", "cfb", "cfp", "cfd", "cfpb", "cfpp", "cfpd"] # fmt: skip
        for type_ in types:
            fd.plot(type_=type_, show=False)
