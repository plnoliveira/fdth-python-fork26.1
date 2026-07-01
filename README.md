# 📊 fdth

A python port of the [fdth](https://github.com/jcfaria/fdth) R library.

## 🇧🇷 (Portuguese) Ainda falta algo?

Este pacote foi desenvolvido como projeto para a disciplina  
**"Probabilidade e Estatística"** (Ciência da Computação, UESC), nos semestres  
de **2025.1 e 2025.2**. Vale ressaltar que o projeto já é anterior a isso,  
porém incorporou a estruturação com foco em **orientação a objetos**  
apenas em 2025.

O projeto já inclui **todas as funcionalidades** do pacote **fdth original**,  
que foi feito para ser usado na linguagem **R**.

## 🧪 examples

**There's an example of the "group by" functionality, which does not supports more than
one argument in the original package:**

## 📊 Example

```python
import pandas as pd
from fdth import fdt
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Altura': [170, 175, 180, 165, 172, 168, 175, 170, 160, 155],
    'Peso': [70, 75, 80, 65, 72, 68, 75, 70, 60, 55],
    'Idade': [30, 20, 30, 18, 30, 27, 24, 50, 60, 9],
    'Sexo': ['M', 'I', 'M', 'I', 'M', 'F', 'I', 'F', 'F', 'F'],
    'Regiao': ['Nordeste', 'Sudeste', 'Nordeste', 'Sudeste', 'Norte', 'Norte', 'Sudeste', 'Nordeste', 'Norte', 'Sudeste'],
    'EstadoCivil': ['Casado', 'Solteiro', 'Solteiro', 'Casado', 'Solteiro', 'Solteiro', 'Casado', 'Casado', 'Solteiro', 'Casado']
})

mfdt_multi = fdt(df, by=['Sexo', 'EstadoCivil', 'Regiao'])

mfdt_multi.plot(numeric_type="fh", categorical_type="fb")
```
Page 1
<img width="1588" height="802" alt="image" src="https://github.com/user-attachments/assets/7dbc6c71-8c40-438f-83dc-83e2ba94849f" />

Page 2
<img width="1599" height="811" alt="image" src="https://github.com/user-attachments/assets/68852616-0e8f-4174-8c92-2fd15f82f320" />


More examples can be found at the `examples/python` folder.

## 🛠️ development

First of all, clone the repository (para um tutorial em português  
disso, olhe [este arquivo](HelpGit.md)).

Then, set up a virtual environment:

```sh
# on linux and windows
python -m venv venv

# on linux
source venv/bin/activate

# on windows (command prompt)
venv\Scripts\activate.bat

# on windows (PowerShell)
Set-ExecutionPolicy Unrestricted -Scope Process
venv\Scripts\activate.ps1
```

Install the package to the venv (needs to be done only once):

```sh
pip install -e .
```

## 🧰 tools

Use `unittest` for running automatic tests (included in python):

```sh
python -m unittest discover -s tests
```

Use `black` for code formatting (`pip install black`):

```sh
black .
```

Use `pdoc` for doc generation (`pip install pdoc`):

```sh
pdoc -o doc fdth
```

Use `mypy` for type checking (`pip install mypy`):

```sh
mypy --strict --cache-fine-grained .
```

## 👥 credits

🔹 [Original version](https://github.com/jcfaria/fdth) created by  
   - [José Cláudio Faria](https://github.com/jcfaria),  
   - [Ivan Bezerra Allaman](https://github.com/ivanalaman)  
   - [Jakson Alves de Aquino](https://github.com/jalvesaq).

🔹 [Initial python port](https://github.com/yuriccosta/fdth-python) by  
   - [Emyle Silva](https://github.com/EmyleSilva),  
   - [Lucas Gabriel Ferreira](https://github.com/lgferreiracic),  
   - [Yuri Coutinho Costa](https://github.com/yuriccosta),  
   - [Maria Clara](https://github.com/MaryClaraSimoes).

🔹 **Previous** version made by  
   - Gabriel Galdino,
   - Luciene Mª Torquato C. Batista,  
   - Stella Ribas,
   - Thainá Guimarães
   - Yohanan Santana.
  
🔹 **Current** version made by  
   - Alex Amaral dos Santos,  
   - Isaque Silva Passos Ribeiro,  
   - Kaiala de Jesus Santos
   - Olinoedson Silva Sena.
