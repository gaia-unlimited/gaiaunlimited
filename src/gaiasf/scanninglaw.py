import pickle
from pathlib import Path

# from time import perf_counter

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.funcs import spherical_to_cartesian
import numpy as np
import pandas as pd
from scipy import spatial

from gaiasf import fetch_utils

__all__ = [
    "make_rotmat",
    "GaiaScanningLaw",
    "obmt2tcbgaia",
    "angle2dist3d",
    "check_gaps",
    "cartesian_to_spherical",
]

pkgdatadir = Path(__file__).parent / "data"


def obmt2tcbgaia(obmt):
    """
    Calculate Gaia Barycenter coordinate time (TCB, days) from OnBoard Mission Time (OBMT, revs).

    Args:
        obmt: OBMT in revs.

    Returns:
        tcb: TCB in days.
    """
    return (obmt - 1717.6256) / 4 - (2455197.5 - 2457023.5 - 0.25)


def make_rotmat(fov1_xyz, fov2_xyz):
    """Make rotational matrix from ICRS to Gaia body frame(ish).

    Args:
    fov1_xyz (array-like): vector pointing to FoV1. Should have shape (..., 3)
    fov2_xyz (array-like): vector pointing to FoV2. Should have shape (..., 3)

    Returns:
        rotation matrix (..., 3, 3)
    """
    fov1_xyz, fov2_xyz = np.atleast_2d(fov1_xyz), np.atleast_2d(fov2_xyz)
    _xaxis = fov1_xyz
    _zaxis = np.cross(fov1_xyz, fov2_xyz)
    _yaxis = -np.cross(_xaxis, _zaxis)
    _yaxis /= np.linalg.norm(_yaxis, axis=1)[:, np.newaxis]
    _zaxis /= np.linalg.norm(_zaxis, axis=1)[:, np.newaxis]
    _matrix = np.moveaxis(np.stack((_xaxis, _yaxis, _zaxis)), 1, 0)
    return np.squeeze(_matrix)


def angle2dist3d(sepangle):
    """Get equivalent 3d distance of an angle on a unit sphere."""
    r = 2 * np.sin(np.deg2rad(sepangle) / 2.0)
    return r


def cartesian_to_spherical(xyz):
    lon = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.rad2deg(
        np.arctan2(xyz[:, 2], np.sqrt(xyz[:, 0] ** 2.0 + xyz[:, 1] ** 2.0))
    )
    return lon, lat


# def spherical_to_cartesian()

# TODO jit
def check_gaps(gaps, x):
    """Check if values of array x falls in any gaps.

    Args:
        gaps (array): 2d array of [n_gaps, 2] defining lower and upper boundary of each gap.
        x (array): values to check

    Returns:
        boolean array: True if outside of any gap False if not.
    """
    cond = np.full(x.shape, False)
    for i in range(gaps.shape[0]):
        cond = cond | ((x > gaps[i, 0]) & (x < gaps[i, 1]))
    return ~cond


version_mapping = {
    "dr3_nominal": {
        "filename": "CommandedScanLaw_001.csv",
        "column_mapping": {"jd_time": "tcb_at_gaia"},
    },
    "dr2_nominal": {
        "filename": "DEOPTSK-1327_Gaia_scanlaw.csv",
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
    "dr2_cog3": {
        "filename": "cog_dr2_scanning_law_v2.csv",
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
}


class GaiaScanningLaw:
    """Initialize a version of Gaia's scanning law.

    Args:
        version (str, optional): Version of the FoV pointing data file to use.
            One of ["dr3_nominal", "dr2_nominal", "dr2_cog3"]. Defaults to "dr3_nominal".
        gaplist (str, optional): Name of the gap list. Defaults to "dr2/Astrometry".

    """

    version_trange = {
        "cogi_2020": [1192.13, 3750.56],
        "cog3_2020": [1192.13, 3750.56],
        "dr2_nominal": [1192.13, 3750.56],
        "dr3_nominal": [1192.13, 5230.09],
    }

    def __init__(self, version="dr3_nominal", gaplist="dr3/Astrometry", **kwargs):

        if version not in version_mapping:
            raise ValueError("Unsupported version")
        self.version = version

        # Get pointing data
        df_path = fetch_utils.get_datadir() / (version + ".pkl")
        if not df_path.exists():
            fetch_utils.download_scanninglaw(version)
        df = pd.read_pickle(df_path).sort_values(by=["tcb_at_gaia"])
        self.pointingdf = df

        self.fov1 = coord.ICRS(
            df["ra_fov1"].values * u.deg, df["dec_fov1"].values * u.deg
        )
        self.fov2 = coord.ICRS(
            df["ra_fov2"].values * u.deg, df["dec_fov2"].values * u.deg
        )
        self.xyz_fov1 = self.fov1.cartesian.xyz.value.T  # (N,3)
        self.xyz_fov2 = self.fov2.cartesian.xyz.value.T  # (N,3)
        self.rotmat = make_rotmat(self.xyz_fov1, self.xyz_fov2)
        self.tcb_at_gaia = df["tcb_at_gaia"].values

        self._setup_fov_trees()

        # this radius guarantees one sample on either side of the FoV front
        # self.r_search = angle2dist3d(0.4 * u.deg)
        self.r_search = np.tan(np.deg2rad(0.35 * np.sqrt(2) + 25.0 / 3600))

        self.gaplist = gaplist
        if gaplist is None:
            self.gaps = np.empty((0, 2))
        else:
            gaplist_path = pkgdatadir / (gaplist + ".csv")
            if not gaplist_path.exists():
                raise ValueError("gaplist is not valid.")
            self.gaps = pd.read_csv(gaplist_path).iloc[:, :2].to_numpy()
            self.gaps = obmt2tcbgaia(self.gaps)
            self.gaps = np.concatenate(
                [
                    [[-np.inf, obmt2tcbgaia(self.version_trange[self.version][0])]],
                    self.gaps,
                    [[obmt2tcbgaia(self.version_trange[self.version][1]), np.inf]],
                ]
            )

    def __repr__(self):
        return f"GaiaScanningLaw(version='{self.version}', gaplist='{self.gaplist}')"

    def _setup_fov_trees(self):
        # make kdtrees of pre-shifted FoV start locations and
        # cache them to local disk
        tree_cache_path = fetch_utils.get_datadir() / (f"{self.version}-cached-trees.p")
        if tree_cache_path.exists():
            with open(tree_cache_path, "rb") as f:
                self.tree_fov1, self.tree_fov2 = pickle.load(f)
        else:
            # offset_ac = 221 * u.arcsec
            # offset_al = -0.5 * u.deg
            offset_ac = 0 * u.arcsec
            offset_al = 0 * u.deg
            basic_angle = 106.5 * u.deg

            x, y, z = coord.spherical_to_cartesian(1, offset_ac, offset_al)
            xyz_fov1_body = np.array([x.value, y.value, z.value])
            xyz_fov1 = np.einsum("nij,i->nj", self.rotmat, xyz_fov1_body)
            tree_fov1 = spatial.KDTree(xyz_fov1)

            x, y, z = coord.spherical_to_cartesian(
                1, -offset_ac, basic_angle + offset_al
            )
            xyz_fov2_body = np.array([x.value, y.value, z.value])
            xyz_fov2 = np.einsum("nij,i->nj", self.rotmat, xyz_fov2_body)
            tree_fov2 = spatial.KDTree(xyz_fov2)
            print(f"Writing cached kdtree to {tree_cache_path}")
            with open(tree_cache_path, "wb") as f:
                pickle.dump((tree_fov1, tree_fov2), f)
            self.tree_fov1, self.tree_fov2 = tree_fov1, tree_fov2

    def query(self, ra_deg, dec_deg, count_only=False):
        """Query the scanning law for a given position.

        Returned times are in Julian days in TCB with time origin
        2010-01-01:T00:00 (JD 2455197.5). See `this page`_ for details.

        .. _this page: https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_auxiliary_tables/ssec_dm_commanded_scan_law.html

        Args:
            ra_deg (float): right ascension in degrees
            dec_deg (float): declination in degrees
            count_only (bool): only return the total number of scans.

        Returns
            [fov1_times, fov2_times]
            [n_fov1, n_fov2] if count_only=True
        """
        query_coord_xyz = spherical_to_cartesian(
            1.0, np.deg2rad(dec_deg), np.deg2rad(ra_deg)
        )

        times = []
        for tree_fov, offset, lon0 in zip(
            [self.tree_fov1, self.tree_fov2],
            [221 / 3600.0, -221 / 3600.0],
            [0.0, 106.5],
        ):
            # first drastically cutdown snapshots to check by positional matching
            tidx = tree_fov.query_ball_point(
                query_coord_xyz, self.r_search, return_sorted=True
            )
            if len(tidx) == 0:
                return
            tidx = np.array(tidx)
            # check if actually in FoV and get the first time indices of each scan
            # TODO: aberration
            query_coord_xyz_body_t = np.einsum(
                "nij,j->ni", self.rotmat[tidx], query_coord_xyz
            )
            lon, lat = cartesian_to_spherical(query_coord_xyz_body_t)
            # fovcond = (np.abs(lat - offset) < 0.35) & (lon > lon0)
            fovcond = (np.abs(lat - offset) < 0.35) & (np.abs(lon - lon0) < 1.0)
            tidx_in = tidx[fovcond]
            # scanning law is sampled 10s and each scan takes ~6 hrs (21600s)
            tidx_scan = tidx_in[np.insert(tidx_in[1:] - tidx_in[:-1] > 360, 0, True)]
            t_scan = self.tcb_at_gaia[tidx_scan]
            # TODO: interpolation here
            # check if times fall in any gaps
            cond_gaps = check_gaps(self.gaps, t_scan)
            t_scan = t_scan[cond_gaps]

            times.append(t_scan)

        if count_only:
            return [len(times[0]), len(times[1])]
        return times
