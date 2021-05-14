from setuptools import setup, find_packages

# TODO: find a way to include the neuroshare package from github
# TODO: update requirements.txt
version_reqs = ['pandas',
                'scipy',
                'matplotlib',
                'seaborn',
                'numba',
                'pytest',
                'tdt',
                'mne',
                'ipywidgets',
                'cachey',
                'cached_property',
                'dask[complete]',
                'ipympl',
                'sphinx-rtd-theme',
                'h5py',
                'xlrd',
                'openpyxl']

setup(
    name="eba_toolkit",
    version="0.0",
    author="James Trevathan & Stephan Blanz",
    author_email="james.trevathan@gmail.com, stephan.l.blanz@gmail.com",
    packages=find_packages(),
    # install_requires=version_reqs
)
