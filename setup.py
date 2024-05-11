from setuptools import setup, find_packages

# TODO: find a way to include the neuroshare package from github
# TODO: update requirements.txt
version_reqs = ['pandas<2.0.0',
                'numpy<1.21,>=1.17',
                'scipy==1.10',
                'matplotlib<3.5.0',
                'seaborn',
                'numba==0.56.4',
                'pytest',
                'tdt',
                'mne==1.5.1',
                'ipywidgets',
                'cachey',
                'cached_property',
                'dask[complete]==2022.12.1',
                'ipympl',
                'sphinx-rtd-theme',
                'h5py',
                'xlrd', # Neeed to remove this excel dependency
                'openpyxl',
                'pillow>=7.1.0',
                'cycler>=0.10',
                'tqdm',
                'plotly==5.15.0', 
                'plotly_resampler==0.8.3.2'
                'dash==2.14.2'
                'dash-bootstrap-components==1.5.0'
                'dash-extensions==1.0.7'
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
