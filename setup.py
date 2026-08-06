from setuptools import setup, find_packages

# TODO: find a way to include the neuroshare package from github
# TODO: update requirements.txt
version_reqs = ['pandas',
                'numpy',
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
                'xlrd', # Need to remove this Excel dependency
                'openpyxl',
                'pillow>=7.1.0',
                'cycler>=0.10',
                'tqdm', # Only used in example notebooks at this point
                ]

setup(
    name="pyeCAP",
    version="0.0.1",
    author="James Trevathan & Stephan Blanz & Matthew Laluzerne",
    author_email="james.trevathan@gmail.com, stephan.l.blanz@gmail.com",
    packages=find_packages(),
    install_requires=version_reqs,
    url="https://github.com/ludwig-lab/pyeCAP",
    download_url="https://github.com/ludwig-lab/pyeCAP/archive/refs/tags/v_0.0.1.tar.gz"   # v_0.0 is not stable
)
