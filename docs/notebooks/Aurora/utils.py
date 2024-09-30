
import os
import requests
from tqdm import tqdm

import numpy as np
import pandas as pd

import astropy.units as u
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, Galactocentric

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def download_file(url, file_name):

    print("Starting download from", url)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        if os.path.exists(file_name):
            print(f"File '{file_name}' already downloaded")
        else:
            with open(file_name, 'wb') as f:
                pbar = tqdm()
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
    print("Done")



def plot_hpxmap(sky, frame='icrs', step=0.5, norm='log', vmin=None, vmax=None,
                xlabel=True, ylabel=True, title=None, cmap='inferno', cbar=True, ax=None):

    npix = sky.shape[0]
    nside = int(np.sqrt(npix // 12))

    l_grid = np.arange(-180.0, 180.0, step)
    b_grid = np.arange(-90.0, 90.0, step)
    b_mesh, l_mesh = np.meshgrid(b_grid, l_grid)

    hp = HEALPix(nside=nside, order='nested', frame=frame)
    co = SkyCoord(l_mesh, b_mesh, frame='galactic', unit='deg')

    prb_mesh = hp.interpolate_bilinear_skycoord(co, sky)

    if ax is None:
        ax = plt.gca()

    if norm == 'log':
        if vmin is None:
            vmin = np.nanmin(sky)
        if vmax is None:
            vmax = np.nanmax(sky)
        if vmin == 0:
            vmin = 1e-6*vmax
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(l_grid, b_grid, prb_mesh.T,
                           cmap=cmap, norm=norm, rasterized=True)
    else:
        im = ax.pcolormesh(l_grid, b_grid, prb_mesh.T,
                           cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, rasterized=True)

    ax.set_xlim(180.0, -180.0)
    ax.set_ylim(-90, 90)
    if cbar:
        plt.colorbar(im, ax=ax)

    ax.set_title(title)
    if xlabel is True:
        ax.set_xlabel(r"$l$ [deg]")
    else:
        ax.set_xlabel(xlabel)
    if ylabel is True:
        ax.set_ylabel(r"$b$ [deg]")
    else:
        ax.set_ylabel(ylabel)

    return im



def plot_projections(df, xlim=None, ylim=None, zlim=None, bins=200, cmap='inferno'):

    x_sun = - Galactocentric().galcen_distance.to('kpc').value
    y_sun = 0.0
    z_sun = Galactocentric().z_sun.to('kpc').value

    plt.rc('font', size=6.0)
    inch = 2.54  ## cm
    width, height = 17/inch, 4/inch
    plt.figure(figsize=(width, height), layout='constrained')

    ax = plt.subplot(1, 3, 1)
    l = (df['l'] + 180) % 360 - 180
    b = df['b']
    df_hist_lb, l_bins, b_bins = np.histogram2d(l, b, bins=bins)
    im = ax.pcolormesh(l_bins, b_bins, df_hist_lb.T, cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ##
    plt.xlabel("l [deg]")
    plt.xlim(180, -180)
    plt.ylabel("b [deg]")

    ax = plt.subplot(1, 3, 2)
    x = df['x']
    y = df['y']
    df_hist_xy, x_bins, y_bins = np.histogram2d(x, y, bins=bins)
    im = ax.pcolormesh(x_bins, y_bins, df_hist_xy.T, cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='w')
    ##
    plt.xlabel("x [kpc]")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylabel("y [kpc]")
    if ylim is not None:
        ax.set_ylim(ylim)

    ax = plt.subplot(1, 3, 3)
    x = df['x']
    z = df['z']
    df_hist_xz, x_bins, z_bins = np.histogram2d(x, z, bins=bins)
    im = ax.pcolormesh(x_bins, z_bins, df_hist_xz.T,
                       cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='w')
    ##
    plt.xlabel("x [kpc]")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylabel("z [kpc]")
    if ylim is not None:
        ax.set_ylim(zlim)



def plot_final_pane(df, dF, dom, proj,
                    xlim=None, ylim=None, zlim=None, bins=200, figsize=(9, 6), cmap='inferno'):

    x_sun = - Galactocentric().galcen_distance.to('kpc').value
    y_sun = 0.0
    z_sun = Galactocentric().z_sun.to('kpc').value

    plt.rc('font', size=6.0)
    plt.figure(figsize=figsize, layout='constrained')

    #
    # lb

    ax = plt.subplot(3, 3, 1)
    l = (df['l'] + 180) % 360 - 180
    b = df['b']
    df_hist_lb, l_bins, b_bins = np.histogram2d(l, b, bins=bins)
    im = ax.pcolormesh(l_bins, b_bins, df_hist_lb.T,
                       cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.ylabel(r"$b$ [deg]")

    ax = plt.subplot(3, 3, 4)
    l = (dom.l + 180) % 360 - 180
    b = dom.b
    dF_hist_lb, _, _ = np.histogram2d(l, b, bins=[l_bins, b_bins], weights=dF.sum(axis=1))
    im = ax.pcolormesh(l_bins, b_bins, dF_hist_lb.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.ylabel(r"$b$ [deg]")

    ax = plt.subplot(3, 3, 7)
    diff_lb = (dF_hist_lb - df_hist_lb) / (dF_hist_lb + df_hist_lb)
    im = ax.pcolormesh(l_bins, b_bins, diff_lb.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    plt.xlim(180, -180)
    plt.xlabel(r"$l$ [deg]")
    plt.ylabel(r"$b$ [deg]")

    #
    # xy

    xy_edges = [bins, bins]
    if xlim is not None:
        xy_edges[0] = np.linspace(xlim[0], xlim[1], bins+1)
    if ylim is not None:
        xy_edges[1] = np.linspace(ylim[0], ylim[1], bins+1)

    ax = plt.subplot(3, 3, 2)
    x = df['x']
    y = df['y']
    df_hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=xy_edges)
    im = ax.pcolormesh(x_edges, y_edges, df_hist_xy.T,
                       cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='w')
    ax.set_ylabel(r"$y$ [kpc]")

    ax = plt.subplot(3, 3, 5)
    x = proj.xyz[0].ravel()
    y = proj.xyz[1].ravel()
    dF_hist_xy, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=dF.ravel())
    im = ax.pcolormesh(x_edges, y_edges, dF_hist_xy.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='w')
    ax.set_ylabel(r"$y$ [kpc]")

    ax = plt.subplot(3, 3, 8)
    diff_xy = (dF_hist_xy - df_hist_xy) / (dF_hist_xy + df_hist_xy)
    im = ax.pcolormesh(x_edges, y_edges, diff_xy.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [y_sun], '*', markersize=5, c='w')
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$y$ [kpc]")

    #
    # xz

    xz_edges = [bins, bins]
    if xlim is not None:
        xz_edges[0] = np.linspace(xlim[0], xlim[1], bins+1)
    if zlim is not None:
        xz_edges[1] = np.linspace(zlim[0], zlim[1], bins+1)

    ax = plt.subplot(3, 3, 3)
    x = df['x']
    z = df['z']
    df_hist_xz, x_edges, z_edges = np.histogram2d(x, z, bins=xz_edges)
    im = ax.pcolormesh(x_edges, z_edges, df_hist_xz.T,
                       cmap=cmap, norm='log', rasterized=True)
    cb = plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='w')
    ax.set_ylabel(r"$z$ [kpc]")

    ax = plt.subplot(3, 3, 6)
    x = proj.xyz[0].ravel()
    z = proj.xyz[2].ravel()
    dF_hist_xz, _, _ = np.histogram2d(x, z, bins=(x_edges, z_edges), weights=dF.ravel())
    im = ax.pcolormesh(x_edges, z_edges, dF_hist_xz.T,
                       cmap=cmap, norm=LogNorm(vmin=cb.vmin, vmax=cb.vmax), rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='w')
    ax.set_ylabel(r"$z$ [kpc]")

    ax = plt.subplot(3, 3, 9)
    diff_xz = (dF_hist_xz - df_hist_xz) / (dF_hist_xz + df_hist_xz)
    im = ax.pcolormesh(x_edges, z_edges, diff_xz.T, cmap='bwr', rasterized=True)
    plt.colorbar(im, ax=ax)
    ax.plot([x_sun], [z_sun], '*', markersize=5, c='w')
    ax.set_xlabel(r"$x$ [kpc]")
    ax.set_ylabel(r"$z$ [kpc]")
