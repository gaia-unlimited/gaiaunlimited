import ftplib
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
    """Get gaiasf data directory as Path.

    Return type:
        pathlib.PosixPath object
    """
    p = (
        Path(os.getenv("GAIAUNLIMITED_DATADIR", "~/.gaiaunlimited"))
        .expanduser()
        .resolve()
    )
    return p


# adapted from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url, file, desc=None, chunk_size=1024, md5sum=None):
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


def download_ftp(url, dest_file, desc=None, chunk_size=1024, md5sum=None):
    """Download file from an FTP url.

    Args:
        url (str): url string
        dest_file (file object): destination file object to write the content to.
        desc (str, optional): Description of progressbar. Defaults to None.
        chunk_size (int, optional): Chunk size to iteratively update progrss and md5sum. Defaults to 1024.
        md5sum (str, optional): The expected md5sum to check against. Defaults to None.

    Raises:
        DownloadError: raised when md5sum differs.
        ValueError: raised when the input url is invalid
    """
    url_bits = [x for x in url.split("/") if x]
    if url_bits[0].startswith("ftp"):
        FTP_Class = ftplib.FTP
    elif url_bits[0].startswith("sftp"):
        FTP_Class = ftplib.FTP_TLS
    else:
        raise ValueError("invalid url for FTP download")

    auth_bits = url_bits[1].split("@")
    if len(auth_bits) == 1:
        user = ""
        passwd = ""
        host = auth_bits[0]
    else:
        user, passwd = auth_bits[0].split(":")
        host = auth_bits[1]

    path = "/".join(url_bits[2:-1])
    filename = url_bits[-1]

    if desc is None:
        desc = filename

    if len(path) == 0 or len(filename) == 0:
        raise ValueError(
            "failed to parse input url as a valid server, path, and filename"
        )

    with FTP_Class(host=host, user=user, passwd=passwd) as ftp:
        ftp.prot_p()
        ftp.cwd(path)
        total = ftp.size(filename)
        sig = hashlib.md5()

        with io.BytesIO() as rawfile:
            with tqdm(
                desc=desc,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:

                def tqdm_callback(data):
                    bar.update(len(data))
                    rawfile.write(data)

                ftp.retrbinary(f"RETR {filename}", tqdm_callback)

            if md5sum:
                if sig.hexdigest() != md5sum:
                    raise DownloadError(
                        "The MD5 sum of the downloaded file is incorrect.\n"
                        + f"  download: {sig.hexdigest()}\n"
                        + f"  expected: {md5sum}\n"
                    )

            rawfile.seek(0)
            if filename.endswith("gz"):
                with gzip.open(rawfile) as tmp:
                    shutil.copyfileobj(tmp, dest_file)
            else:
                shutil.copyfileobj(rawfile, dest_file)

    dest_file.seek(0)


scanlaw_datafiles = {
    "dr2_cog3": {
        "url": "https://zenodo.org/record/8300616/files/cog_dr2_scanning_law_v2.csv",
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
    "full_operational_mission": {
        "url": "sftp://anonymous@ftp.cosmos.esa.int/GAIA_PUBLIC_DATA/GaiaScanningLaw/FullGaiaMissionScanningLaw/commanded_scan_law.csv.gz",
        "md5sum": "36a30f236da021b10573917ccba6d89a",
        "column_mapping": {"jd_time": "tcb_at_gaia"},
    },
}


def download_scanninglaw(name):
    """
    Download scanning law datafiles if it does not already exist.

    This function downloads and normalizes column names of each data file
    and saves the resulting pandas.DataFrame as pickle in ``GAIASF_DATADIR``.

    Args:
        name(str) : scanning law name. One of ["dr2_cog3", "dr3_nominal"] or "all"
            to download everything.
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
            if item["url"].startswith("ftp") or item["url"].startswith("sftp"):
                download_ftp(item["url"], f, md5sum=item["md5sum"], desc=desc)
            else:
                download(item["url"], f, md5sum=item["md5sum"], desc=desc)
            df = pd.read_csv(f).rename(columns=item["column_mapping"])
            savedir.mkdir(exist_ok=True)
            df.to_pickle(savepath)


class DownloadMixin:
    """Mixin for downloading data file(s) from a url to GAIASF_DATADIR.

    The data locations should be specified as a dictionary of (filename, url) key-value pairs.
    Then, use ``_get_data`` method to retrieve the full Path to the downloaded file.

    Example:
        class SomeClassUsingLargeDatafile(DownloadMixin):
            datafiles = {'myfile1': "http://url/to/myfile1", 'myfile2': 'http://url/to/myfile2'}

            def __init__(self, *args, **kwargs):
                ...
                # If the file does not exist in GAIASF_DATADIR, the following line will
                # download it from the url.
                path_to_file1 = self._get_data('myfile1')
                with h5py.File(path_to_file1) as f:
                    ...
    """

    def _get_data(self, filename):
        """Download data files specified in datafiles dict class attribute."""
        savedir = get_datadir()
        if not savedir.exists():
            print("Creating directory", savedir)
            os.makedirs(savedir)
        fullpath = get_datadir() / filename
        if not fullpath.exists():
            url = self.datafiles[filename]
            with open(fullpath, "wb") as f:
                download(url, f)
        return fullpath
