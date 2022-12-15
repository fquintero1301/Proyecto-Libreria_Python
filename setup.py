from setuptools import setup
from pathlib import Path

# cuál nombre de app??, y autores??
setup(
    name= 'datascienceproyect',
    version= '0.0.1',
    license= 'MIT',
    description= 'Librería para limpieza y visualización de datos, y Machine learning',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author= 'Clase Data Science The Bridge Septiembre 2022',
    url= 'https://github.com/LLBF/Proyecto_libreria_DS',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        ],
    python_requires= '>=3.7'
)