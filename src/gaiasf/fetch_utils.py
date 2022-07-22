import gzip
import hashlib
import io
import os
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

__all__ = ["download", "download_scanninglaw", "get_datadir", "DownloadMixin"]


class DownloadError(Exception):
    """
    An exception that occurs while trying to download a file.
    """


def get_datadir():
    p = Path(os.getenv("GAIASF_DATADIR", "~/.gaiasf")).expanduser().resolve()
    return p


# adapted from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url: str, file, desc=None, chunk_size=1024, md5sum=None):
    """Download file from a url.

    Args:
        url (str): url string
        file (file object): file object to write the content to.
        desc (str, optional): Description of progressbar. Defaults to None.
        chunk_size (int, optional): Chunk size to iteratively update progrss and md5sum. Defaults to 1024.
        md5sum (str, optional): The expected md5sum to check against. Defaults to None.

    Raises:
        DownloadError: raised when md5sum differs.
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    sig = hashlib.md5()
    filename_from_url = url.split("/")[-1]
    if desc is None:
        desc = filename_from_url
    with io.BytesIO() as rawfile:
        with tqdm(
            desc=desc,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = rawfile.write(data)
                sig.update(data)
                bar.update(size)

        if md5sum:
            if sig.hexdigest() != md5sum:
                raise DownloadError(
                    "The MD5 sum of the downloaded file is incorrect.\n"
                    + "  download: {}\n".format(sig.hexdigest())
                    + "  expected: {}\n".format(md5sum)
                )

        rawfile.seek(0)
        if filename_from_url.endswith("gz"):
            with gzip.open(rawfile) as tmp:
                shutil.copyfileobj(tmp, file)
        else:
            shutil.copyfileobj(rawfile, file)

    file.seek(0)


scanlaw_datafiles = {
    "dr2_cog3": {
        "url": "https://dataverse.harvard.edu/api/access/datafile/4180696",
        "md5sum": "b4dc9cd9f3a6930c38a19a3535eaeb86",
        "column_mapping": {
            "JulianDayNumberRefEpoch2010TCB@Gaia": "tcb_at_gaia",
            "JulianDayNumberRefEpoch2010TCB@Barycentre_1": "bjd_fov1",
            "JulianDayNumberRefEpoch2010TCB@Barycentre_2": "bjd_fov1",
            "ra_FOV_1(deg)": "ra_fov1",
            "dec_FOV_1(deg)": "dec_fov1",
            "scanPositionAngle_FOV_1(deg)": "scan_angle_fov1",
            "ra_FOV_2(deg)": "ra_fov2",
            "dec_FOV_2(deg)": "dec_fov2",
            "scanPositionAngle_FOV_2(deg)": "scan_angle_fov2",
        },
    },
    "dr3_nominal": {
        "url": "http://cdn.gea.esac.esa.int/Gaia/gedr3/auxiliary/commanded_scan_law/CommandedScanLaw_001.csv.gz",
        "md5sum": "82d24407396f6008a9d72608839533a8",
        "column_mapping": {"jd_time": "tcb_at_gaia"},
    },
}


def download_scanninglaw(name):
    """
    Download scanning law datafiles if it does not already exist.

    This function downloads and normalizes column names of each data file
    and saves the resulting pandas.DataFrame as pickle.

    Output directory is ~/.gaia_scanninglaw by default but can be set with
    the environment variable GAIA_SCANNINGLAW_DATADIR.

    Args:
        name(str) : scanning law name
    """
    if name not in scanlaw_datafiles and name != "all":
        raise ValueError(
            "{name} is not a valid scanning law name; should be one of {names}".format(
                name=name, names=scanlaw_datafiles.keys()
            )
        )

    if name == "all":
        for k in scanlaw_datafiles.keys():
            download_scanninglaw(k)
    else:
        item = scanlaw_datafiles[name]
        savedir = get_datadir()
        savepath = savedir / (name + ".pkl")
        if savepath.exists():
            print("{savepath} already exists; doing nothing.".format(savepath=savepath))
            return
        with io.BytesIO() as f:
            desc = "Downloading {name} scanning law file".format(name=name)
            download(item["url"], f, md5sum=item["md5sum"], desc=desc)
            df = pd.read_csv(f).rename(columns=item["column_mapping"])
            savedir.mkdir(exist_ok=True)
            df.to_pickle(savepath)


class DownloadMixin:
    """Mixin for downloading data files."""

    def _get_data(self, filename):
        """Download data files specified in datafiles dict class attribute."""
        fullpath = get_datadir() / filename
        if not fullpath.exists():
            url = self.datafiles[filename]
            with open(fullpath, "wb") as f:
                download(url, f)
        return fullpath
