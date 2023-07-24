# neuro base class imports
from .base.ts_data import _TsData
from .base.utils.base import _is_iterable

import sys, os
import requests
import zipfile


def download_data(url, filename, directory=os.getcwd()):
    # warning: make sure that the url links to raw data, otherwise this function will download in an unreadable format.
    """
    Downloads data from a web page in binary format and saves it under a specified file name and directory.

    Parameters
    ----------
    url : str
        Url of the web page to download from.
    filename : str
        Name of the file (with extension) to save under.
    directory : str
        Directory to save the file in. Defaults to saving in the current directory.

    Returns
    -------
    str
        Absolute path name of downloaded file.
    """
    r = requests.get(url)
    with open(complete_pathname, "wb") as f:
        f.write(r.content)
    complete_pathname = os.path.join(directory, filename)
    if complete_pathname.endswith(".zip"):
        with zipfile.ZipFile(complete_pathname, "r") as zip_ref:
            zip_ref.extractall(complete_pathname[:-4])
        return complete_pathname[:-4]
    else:
        return complete_pathname


def concatenate(input_data):
    # TODO: handle this in a better way that works across all classes
    # Handles _TsData
    if _is_iterable(input_data, _TsData):
        data = [item for d in input_data for item in d.data]
        metadata = [item for d in input_data for item in d.metadata]
        chunks = [item for d in input_data for item in d.chunks]
        return type(input_data[0])(data, metadata, chunks=chunks, daskify=False)
    else:
        if _is_iterable(input):
            raise NotImplementedError(
                'pyCAP.concatenate not yet implemented for data type "'
                + str(type(input_data[0]))
                + '"'
            )
        else:
            raise ValueError("Input expected to be an iterable of pyCAP data classes")


class hide_print:
    def __enter__(self):
        self.out = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, a, b, c):
        sys.stdout.close()
        sys.stdout = self.out
