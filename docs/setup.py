from setuptools import setup, find_packages

version_reqs = ["pandas==1.2.2",
                "scipy==1.6.1",
                "matplotlib==3.3.4",
                "seaborn==0.11.1",
                "numba==0.52.0",
                "pytest==6.2.2",
                "tdt==0.4.3",
                "mne=0.22.0",
                "ipywidgets==7.6.3",
                "cachey==0.2.1",
                "cached_property==1.5.2",
                "dask[complete]==2021.2.0",
                "ipympl==0.6.3",
                "sphinx-rtd-theme==0.5.1",
                "h5py==3.1.0",
                "xlrd==2.0.1",
                "openpyxl==3.0.6"]

setup(
    name="eba_toolkit",
    version="0.0",
    author="James Trevathan & Stephan Blanz",
    author_email="james.trevathan@gmail.com, stephan.l.blanz@gmail.com",
    packages=find_packages(),
    # install_requires=version_reqs
)