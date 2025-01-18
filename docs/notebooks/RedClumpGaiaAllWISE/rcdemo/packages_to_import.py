import os, sys, time, math
import agama
import importlib as imp
from time import sleep
import ebf
import numpy, numpy as np
import numpy.polynomial.polynomial as nppoly 
import pickle
import scipy
from scipy import stats
from scipy import integrate
from scipy.stats import norm 
from scipy.stats import poisson
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import scipy.interpolate
import scipy.spatial.qhull as qhull
import astropy, astropy.convolution
from astropy.table import Table, Column
from astropy.io import ascii				
from astroquery.gaia import Gaia																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																												
import colorcet as cc
import natsort
from natsort import natsorted, ns
import itertools
from multiprocessing import Pool
from functools import partial 
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib  as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.ticker import MaxNLocator
import matplotlib.collections
from matplotlib_venn import venn2
from matplotlib.patches import Circle		
# import cmasher as cmr
import pylab
import healpy, healpy as hp
from healpy.newvisufunc import projview, newprojplot
import pandas as pd
import h5py 
import uncertainties.unumpy as unumpy
from configparser import ConfigParser

# from rcdemo import sutil_sanj as sutil
from rcdemo import sutil
from rcdemo import autil







class stopwatch(object):
	# import time
	def __init__(self):		
		self.purpose = 'time'
	
	def begin(self):
		self.t1=time.time()
		
		
	def end(self):
		tm = (time.time()-self.t1)		

		print('')
		print('---------------------------------------------')
		
		print ('time (seconds, minutes, hours) =', str(np.round(tm,2))+','+str(np.round(tm/60.,2))+','+str(np.round(tm/3600.,2)))			

		return tm



timeit = stopwatch()
timeit.begin()


#########################################
# data projection functions


def densprof_indirect(_r,dmodel,count_unc_fac=1,bins=None,suffix='',rmin1=5,rmax1=30,zmin1=-5,
					 zmax1=5,nrbins1=40,hresz=0.5,svit=False,svnam='',_z=[],zmode=False,zphimode=False,
					 tempdir='',plotit=False,prntsv=False):
	
	# add phi bins
			
	'''
	integrate over dr & plot
	'''




	# plt.close()
	if zmode == False:
		
		rmins=np.linspace(rmin1,rmax1,nrbins1)
		bval = []
		rcen  = []
		nfull = []
		
	
		for inum,rmin in enumerate(rmins):	
		
			if inum < len(rmins)-1:
				rmax = rmins[inum+1]
				indx=np.where((_r>rmin)&(_r<rmax))[0]
				rcen.append((rmin+rmax)*0.5)
				
				bval.append((dmodel[indx]).sum())
				indfull = np.where(dmodel[indx] > 0.)[0]
				nfull.append(indfull)
	
		

		# print(type(nfull))
		# dtools.picklewrite(nfull,'rubbish',desktop)
		# nfull = np.array(nfull)
		
		# dtools.picklewrite(nfull,'nfull',desktop)
		bval = np.array(bval)
		rcen = np.array(rcen)
		
		wid= (rmax1-rmin1)/float(nrbins1)
		divfac = (2.*np.pi*rcen*wid)
	
		if plotit:
			plt.plot(rcen,bval/divfac,'-',label=suffix)		
			plt.legend()		
			plt.yscale('log')		
		
		
		prof_ = {}
		prof_['rcen'] = rcen
		prof_['ndens'] = bval/divfac
		
		if svit ==True:
			# print('saving plot..')
			dtools.picklewrite(prof_,'profile_'+svnam,tempdir,prnt=prntsv)
		
		
		return bval,rcen	
			
	if zmode == True:
		

		nzbins1 = int((zmax1 - zmin1)/hresz)
		
		# print('checking...')
		# print(nzbins1)		
		
		zmins=np.linspace(zmin1,zmax1,nzbins1) 
		rmins=np.linspace(rmin1,rmax1,nrbins1)
		

	
		storeval = {}
		storenval = {}
		storeuncval = {}
		for inum,rmin in enumerate(rmins):	
		
			if inum < len(rmins)-1:
				rmax = rmins[inum+1]
				rcenval = (rmin+rmax)*0.5
				
				
				
				storeval[rcenval] = {} 
				storenval[rcenval] = {} 
				for inumz,zmin in enumerate(zmins):

					if inumz < len(zmins)-1:
						zmax = zmins[inumz+1]
						indx = np.where((_r>rmin)&(_r<rmax)&(_z>zmin)&(_z<zmax))[0]
						zcenval = (zmin+zmax)*0.5
	
	
						wid= (rmax-rmin)/float(nrbins1)
						divfac = (2.*np.pi*rcenval*wid) #*(zmax - zmin)
	
						
						bval= (((dmodel[indx]).sum()))/divfac
						# bval_unc= (((count_unc_fac[indx]*dmodel[indx]).sum()))/divfac
						# bval_unc= count_unc_fac[indx]*dmodel[indx]
						nval = indx.size

						if indx.size > 0 :
							
						
							storeval[rcenval][zcenval] = bval
							# storeuncval[rcenval][zcenval] = bval_unc
							storenval[rcenval][zcenval] = nval


						else:
							storeval[rcenval][zcenval] = 0.
							# storeuncval[rcenval][zcenval] = bval_unc
							storenval[rcenval][zcenval] = 0.


		storeval1 = storeval.copy()
		storenval1 = storenval.copy()
		dictnames = storeval.keys()
		for ky in dictnames:
			# print(ky)
			tst = list(storeval1[ky].values())
			# print(tst)
			# print(np.sum(tst))
			if np.sum(tst) == 0.:
				del storeval1[ky]
				del storenval1[ky]
	

			
		if svit ==True:
			# print('saving plot..')
			# dtools.picklewrite(prof_,'profile_z_'+svnam,desktop)
			# dtools.picklewrite(storeval,'testprofz',tempdir)			


			dtools.picklewrite(storeval1,'profilez_'+svnam,tempdir,prnt=prntsv)
			dtools.picklewrite(storenval1,'n_profilez_'+svnam,tempdir,prnt=prntsv)
			dtools.picklewrite(storeuncval,'unc_profilez_'+svnam,tempdir,prnt=prntsv)
		
		return	
	
	if zphimode == True:
		

		nzbins1 = int((zmax1 - zmin1)/hresz)
		
		# print('checking...')
		# print(nzbins1)		
		
		phimins=np.linspace(phimin1,phimax1,nphibins1) 
		zmins=np.linspace(zmin1,zmax1,nzbins1) 
		rmins=np.linspace(rmin1,rmax1,nrbins1)
		

	
		storeval = {}
		storenval = {}
		storeuncval = {}
		for inum,rmin in enumerate(rmins):	
		
			if inum < len(rmins)-1:
				rmax = rmins[inum+1]
				rcenval = (rmin+rmax)*0.5
				
				
				
				storeval[rcenval] = {} 
				storenval[rcenval] = {} 
				for inumz,zmin in enumerate(zmins):

					if inumz < len(zmins)-1:
						zmax = zmins[inumz+1]
						indx = np.where((_r>rmin)&(_r<rmax)&(_z>zmin)&(_z<zmax))[0]
						zcenval = (zmin+zmax)*0.5
	
	
						wid= (rmax-rmin)/float(nrbins1)
						divfac = (2.*np.pi*rcenval*wid) #*(zmax - zmin)
	
						
						bval= (((dmodel[indx]).sum()))/divfac
						# bval_unc= (((count_unc_fac[indx]*dmodel[indx]).sum()))/divfac
						# bval_unc= count_unc_fac[indx]*dmodel[indx]
						nval = indx.size

						if indx.size > 0 :
						
							storeval[rcenval][zcenval] = bval
							# storeuncval[rcenval][zcenval] = bval_unc
							storenval[rcenval][zcenval] = nval
		
		
			
		if svit ==True:
			# print('saving plot..')
			# dtools.picklewrite(prof_,'profile_z_'+svnam,desktop)
			# dtools.picklewrite(storeval,'testprofz',tempdir)			


			dtools.picklewrite(storeval,'profilez_'+svnam,tempdir,prnt=prntsv)
			dtools.picklewrite(storenval,'n_profilez_'+svnam,tempdir,prnt=prntsv)
			dtools.picklewrite(storeuncval,'unc_profilez_'+svnam,tempdir,prnt=prntsv)
		
		return	
	
def getdthist(hx,hxdata,mincounts=0):	
	
	'''
	reads in histogram (h1)
	
	'''
	
	xsun = dtools.get_lsr()['xsun']
	zsun = dtools.get_lsr()['zsun']
	
	if len(hx.locations) == 2:	
		print('2D map')

		dx = (hx.locations[0][1:] - hx.locations[0][0:-1])[0]
		dy = (hx.locations[1][1:] - hx.locations[1][0:-1])[0]


		xv,yv = np.meshgrid(hx.locations[0],hx.locations[1],indexing='ij')

		
		dthist={}
		dthist['xv'] = xv.flatten()
		dthist['yv'] = yv.flatten()
		dthist['zv'] = np.zeros(dthist['xv'].size)
		dthist['dx'] = dx
		dthist['dy'] = dy
		dthist['dens'] = hxdata #hx.data
		dthist['ndens'] = hxdata*dx*dy # (hx.data)*dx*dy	
		dthist['rgcd'] = np.sqrt(dthist['xv']**2. + dthist['yv']**2.)
	
	
		l,b,r = autil.xyz2lbr(dthist['xv'] - xsun,dthist['yv'],dthist['zv'] - zsun)
		dthist['l'] = l%360.
		dthist['b'] = b
		dthist['dist'] = r
	
	
	if len(hx.locations) == 3:
		print('3D map')	
		
		dx = (hx.locations[0][1:] - hx.locations[0][0:-1])[0]
		dy = (hx.locations[1][1:] - hx.locations[1][0:-1])[0]
		dz = (hx.locations[2][1:] - hx.locations[2][0:-1])[0]
	
	
		xv,yv,zv = np.meshgrid(hx.locations[0],hx.locations[1],hx.locations[2],indexing='ij')
	
		dthist={}
		dthist['xv'] = xv.flatten()
		dthist['yv'] = yv.flatten()
		dthist['zv'] = zv.flatten()
		dthist['dx'] = dx
		dthist['dy'] = dy
		dthist['dz'] = dz
		dthist['dens'] = hxdata #hx.data
		dthist['ndens'] = hxdata*dx*dy*dz # (hx.data)*dx*dy	
		dthist['rgc'] = np.sqrt(dthist['xv']**2. + dthist['yv']**2.)
		dthist['rgcd'] = np.sqrt(dthist['xv']**2. + dthist['yv']**2.+ dthist['zv']**2.)
		
		l,b,r = autil.xyz2lbr(dthist['xv'] - xsun,dthist['yv'],dthist['zv'] - zsun)
		dthist['l'] = l%360.
		dthist['b'] = b
		dthist['dist'] = r
	
	
	if mincounts > 0:
		
		print('.........')
		print('applying minimum count = '+str(mincounts))
		print('.........')
		
		indmin = np.where(dthist['dens'] < mincounts)[0]
		dthist['dens'][indmin] = 0
		dthist['ndens'][indmin] = 0
	
	return dthist



def cylingrid(h1,add_dust_to_grid_2d=False,add_dust_to_grid_3d=False):
	
	'''
	
	h1
	'''
	
	xsun = dtools.get_lsr()['xsun']

	dthist_use = getdthist(h1,h1.data,mincounts=0)
	dthist_use['rgcv'] = dthist_use['xv'].copy()
	dthist_use['phiv'] = dthist_use['yv'].copy()

		
	dthist_use['drgc'] = dthist_use['dx'].copy()
	dthist_use['dphi'] = dthist_use['dy'].copy()
	

	rmkeys = ['dx','dy','l', 'b', 'dist']
	for ky in rmkeys:
		del dthist_use[ky]
	
	dthist_use['xv'] =dthist_use['rgcv'] *np.cos(np.radians(dthist_use['phiv']))
	dthist_use['yv'] =dthist_use['rgcv'] *np.sin(np.radians(dthist_use['phiv']))
	dthist_use['rgcd'] = dtools.sqrtsum(ds=[dthist_use['xv'],dthist_use['yv'],dthist_use['zv']])

	dthist_use['l'],dthist_use['b'],dthist_use['dist'] = autil.xyz2lbr(dthist_use['xv']-xsun,dthist_use['yv'],dthist_use['zv'])


	dthist_use['l'] = dthist_use['l']%360.

	if add_dust_to_grid_2d:			
		print('2d dust')
		import dustmaps.sfd
		sfd = dustmaps.sfd.SFDQuery()
		coo  = astropy.coordinates.SkyCoord(l=dthist_use['l'] * astropy.units.degree, b=dthist_use['b'] * astropy.units.degree, frame='galactic')
		ebval_ = sfd.query(coo)
		dthist_use['a_g_val']  =  ebval_* ebv2ag_			


	if add_dust_to_grid_3d:
		print('3d dust')
		ebval_ = mydust.getebv(dthist_use['l'].astype(float),dthist_use['b'].astype(float),dthist_use['dist'].astype(float),typ='3d')
		dthist_use['a_g_val'] =  ebval_* ebv2ag_	

			
	
	
	return dthist_use



def densproj_(h1data,typ='xy',normtyp='lognorm',cnt=False,cmap=None,cmapname='viridis',cbar=False,
				lblsuff='',xlbl=True,ylbl=True,cntcolor='black',return_avgcounts=False,plot_avgcounts=False,plot_pixcount=False,
				plot_counts=False,binargs=[],Rwarps=5,vmin=0,vmax=1,delrwarps = 2.,parampack={}):
	
	'''
	binargs=[hbinsx,hbinsy,hbinsz,xmin,xmax,ymin,ymax,zmin,zmax]
	'''
	

	hbinsx,hbinsy,hbinsz,xmin,xmax,ymin,ymax,zmin,zmax = binargs[0],binargs[1],binargs[2],binargs[3],binargs[4],binargs[5],binargs[6],binargs[7],binargs[8]
	
	
	if cmap is not None:	

		cmap = cc.cm[cmap]
	else:
		cmap = cmapname		
		
	nsize = 100
	ddum = {}
	ddum['pxgc'] = np.linspace(xmin,xmax,nsize)
	ddum['pygc'] = np.linspace(ymin,ymax,nsize)		
	ddum['pzgc'] = np.linspace(zmin,zmax,nsize)		


	if typ == 'phiz':
		# raise Exception print('correct for pixels with no data...')
		

		
		ddum['phigc'] = np.linspace(ymin,ymax,nsize)
		
		
		#
		
		Rbin_delta = (xmax - xmin)/hbinsx
		# print(type(Rbin_delta))
		# print(type(Rwarps))
		Rwindx1 = int(Rwarps/Rbin_delta)
		Rwindx2 = int((Rwarps+delrwarps)/Rbin_delta)
		# print(Rwindx1,Rwindx2)
		# print('----')
		
		mymap = h1data.reshape(hbinsx,hbinsy,hbinsz).copy()
		
		valsy = []

		for j in range(hbinsy):	
			vals1 = list(np.zeros(hbinsz))
			for i in range(hbinsx):	

				if i >= Rwindx1 and i <= Rwindx2:	 
					vals1 += mymap[i][j].copy() 
			valsy.append(vals1)
			
		valsy = np.array(valsy)	

		vals = valsy.flatten()		
				
		h2 = sutil.hist_nd([ddum['phigc'],ddum['pzgc']],bins=[hbinsy,hbinsz],normed=False,range=[[ymin,ymax], [zmin,zmax]])
		
		dy = h2.locations[0][1] - h2.locations[0][0]
		dz = h2.locations[1][1] - h2.locations[1][0]
		

		if plot_counts :
		
			h2.data = vals.copy()	
			
			h2.imshow(vals,smooth=False,norm=LogNorm(vmin=vmin,vmax=vmax),cmapname=cmap)							
			# h2.imshow(vals,smooth=False,cmapname=cmap)#,vmin=1e+2,vmax=1e+05)							
			# h2_norm_data = (h2.data)/((h2.data.sum())*dx*dz)		
			# h2.data = h2_norm_data.copy()	
			
			plt.text(0.01,0.9,str(Rwarps)+'<R<'+str(Rwarps+delrwarps),transform=plt.gca().transAxes,fontsize=10,color='black',rotation=0)	
			
			# dtools.profile(flrprof_data['rcen'][inddensgood],flrprof_data['ndens'][inddensgood]/facforobs,range=[[metadata_['rmin1'],metadata_['rmax1']],[densmin,densmax]],bins=10,mincount=1,func=np.median,lw=2,return_profile=True,return_bins=True,fig=False,meanerror=True,style='lines',color='black',label='rub')		
	
	
			
			if xlbl:
				plt.xlabel('$\phi_{GC}$ [kpc]')
			if ylbl:
				plt.ylabel('Z$_{GC}$ [kpc]')
						
			if cnt:	
				# clevels = [0.864,0.39]
				clevels = dtools.get_clevels(dims=2)	
				continf =dtools.clevel(h2,levels=clevels,colors=cntcolor,coords=True)	
			
				return vals,h2,continf
				
			return vals,h2	
		else:	
			return vals,h2

	elif typ == 'Rphi':
		# raise Exception print('correct for pixels with no data...')
		

		ddum['rgc'] = np.linspace(xmin,xmax,nsize)	
		ddum['phigc'] = np.linspace(ymin,ymax,nsize)
		
		bin_incr = hbinsz 
		
		
		i=0
		vals= []
		meanvals= []
		npixvals = []
		tmpvals = []
		while i < h1data.size:
			
			zcell = np.array([j*(parampack['hresz']) + zmin for j in range(bin_incr)])
			# print('not okay...')
			vals.append(h1data[i:i+bin_incr].sum())
			

			'----'
			tmparray = h1data[i:i+bin_incr]
			tmpvals.append(np.nansum(vals))
			
			
			conds = (zcell >= parampack['zcell_min'])&(zcell <= parampack['zcell_max'])		
			if parampack['usemincount']:	
				indfull =  np.where(( tmparray > parampack['mincount'])&conds)[0]
			else:
				indfull =  np.where(conds)[0]
				
			
			  
			meanvals.append(np.mean(h1data[i:i+bin_incr][indfull]))
			# meanvals.append(np.mean(((h1data[i:i+bin_incr])*(zcell))[indfull]))

			npixvals.append(indfull.size)
			'----'
			i+=bin_incr
		
		vals = np.array(vals)	
						
		meanvals = np.array(meanvals)					
		npixvals = np.array(npixvals)								
		tmpvals = np.array(tmpvals)								
		
		h2 = sutil.hist_nd([ddum['rgc'],ddum['phigc']],bins=[hbinsx,hbinsy],normed=False,range=[[xmin,xmax], [ymin,ymax]])
		
		dx = h2.locations[0][1] - h2.locations[0][0]
		dy = h2.locations[1][1] - h2.locations[1][0]

		
		if plot_pixcount :
		
			h2.data = npixvals.copy()				
			if normtyp == 'lognorm':		
				h2.imshow(npixvals,smooth=False,norm=LogNorm(),cmapname=cmap,vmin=1e+2,vmax=1e+05)			
			elif normtyp == '':	
	
				h2.imshow(npixvals,smooth=False,cmapname=cmap)			

			cbarlabel = 'N pixels counted'

		if return_avgcounts :
			
			# print('average counts.')
			h2.data = meanvals.copy()		
			# dtools.picklewrite(h2.data,'tmprub',desktop)		
			
			im1 = 0
			if plot_avgcounts :
				# if normtyp == 'lognorm':		
					# h2.imshow(meanvals,smooth=False,norm=LogNorm(vmin=10,vmax=1e+03),cmapname=cmap)#,vmin=1e+2,vmax=1e+05)			
	
				# if normtyp == '':	
				im1 = h2.imshow(meanvals,smooth=False,cmapname=cmap,vmin=vmin,vmax=vmax)#,vmin=1e+2,vmax=1e+05)			
				# if normtyp == 'lims':	
					# h2.imshow(meanvals,smooth=False,cmapname=cmap,vmin=vmin,vmax=vmax)#,vmin=1e+2,vmax=1e+05)			
	
				cbarlabel = '<'+lblsuff+' > '


			return im1,h2

		if plot_counts :
			h2.data = vals.copy()				

			if normtyp == 'lognorm':		
				# h2.imshow(vals,smooth=False,norm=LogNorm(vmin=10,vmax=1e+03),cmapname=cmap)#,vmin=1e+2,vmax=1e+05)			
				h2.imshow(vals,smooth=False,norm=LogNorm(),cmapname=cmap)#,vmin=1e+2,vmax=1e+05)			

			elif normtyp == '':	
				h2.imshow(vals,smooth=False,cmapname=cmap,vmin=10,vmax=1000)#,vmin=1e+2,vmax=1e+05)			

			cbarlabel = 'N '
			
			print(cbarlabel)
			
			return vals, h2
		else:
			return vals, h2
			

		
		if cbar:
			plt.colorbar(label=cbarlabel)				
		
		# h2_norm_data = (h2.data)/((h2.data.sum())*dx*dy)		
		# h2.data = h2_norm_data.copy()			
		
		if xlbl:
			plt.xlabel('R$_{GC}$ [kpc]')
		if ylbl:	
			plt.ylabel('r$\phi_{GC}$ [kpc]')		
			
					
		
		if cnt:
		
			# clevels = [0.864,0.39]			
			clevels = dtools.get_clevels(dims=2)
	

			continf = dtools.clevel(h2,levels=clevels,colors=cntcolor,coords=True)	
					
			return vals,h2,continf
		else:	
			return vals,h2,tmpvals




			
def data_hist2d_(data,xmin=-40,xmax=40,ymin=-40,ymax=40,
				zmin=-40,zmax=40,hbins=None,hbinsx=10,
				hbinsy=10,hbinsz=10,plt=False,xy=False,xz=False,yz=False,normed=False):
	
	 	
	
	d1 = data.copy()
	
	xbins,ybins,zbins = hbins,hbins,hbins
	if hbins == None:
		xbins,ybins,zbins = hbinsx,hbinsy,hbinsz
		
	print(xbins,ybins,zbins)

	if xy:
		print('xy projection')
		print('xmin,xmax = '+str(xmin)+','+str(xmax))
		print('ymin,ymax = '+str(ymin)+','+str(ymax))

		h1 = sutil.hist_nd([d1['pxgc'],d1['pygc']],bins=[xbins,ybins],normed=normed,range=[[xmin,xmax], [ymin,ymax]])	

	elif xz:
		print('xz projection')		
		print('xmin,xmax = '+str(xmin)+','+str(xmax))
		print('zmin,zmax = '+str(zmin)+','+str(zmax))

		h1 = sutil.hist_nd([d1['pxgc'],d1['pzgc']],bins=[xbins,zbins],normed=normed,range=[[xmin,xmax], [zmin,zmax]])	

	elif yz:
		print('yz projection')		
		print('ymin,ymax = '+str(ymin)+','+str(ymax))
		print('zmin,zmax = '+str(zmin)+','+str(zmax))	
		h1 = sutil.hist_nd([d1['pygc'],d1['pzgc']],bins=[ybins,zbins],normed=normed,range=[[ymin,ymax], [zmin,zmax]])	


	print(h1.data.shape)
	
	
	return h1	
def data_hist3d_(data,xmin=-40,xmax=40,ymin=-40,ymax=40,
				zmin=-40,zmax=40,hbins=None,hbinsx=10,hbinsy=10,hbinsz=10):
	
	d1 = data.copy()
	
	xbins,ybins,zbins = hbins,hbins,hbins
	if hbins == None:
		xbins,ybins,zbins = hbinsx,hbinsy,hbinsz
	

	h1 = sutil.hist_nd([d1['pxgc'],d1['pygc'],d1['pzgc']],bins=[xbins,ybins,zbins],normed=False,range=[[xmin,xmax], [ymin,ymax], [zmin,zmax]])	

			
	return h1

#--------------------------------------------------





def profile(x,z,bins=100,range=None,label=None,mincount=3,fmt='-o',markersize=2,color='',meanerror=False,style=None,func=None,lw=None,return_profile=False,return_bins=False,fig=True):	
	"""
	style: ['lines','fill_between,'errorbar']
	
	To add: bootstrapping errors
	"""


	if lw == None:
		lw=5.0
	#style=['lines','fill_between','errorbar']
	ind=np.where(np.isfinite(x)&np.isfinite(z))[0]
	x=x[ind]
	z=z[ind]
	h=sutil.hist_nd(x,bins=bins,range=range)
	ind1=np.where(h.data>=mincount)[0]
	if func is None:
		ym=h.apply(z,np.median)[ind1]
	else:
		ym=h.apply(z,func)[ind1]
	xm=h.locations[0][ind1]
	yl=h.apply(z,np.percentile,16)[ind1]
	yh=h.apply(z,np.percentile,84)[ind1]

	if meanerror:
		yl=ym+(yl-ym)/np.sqrt(h.data[ind1])
		yh=ym+(yh-ym)/np.sqrt(h.data[ind1])
		

	if fig:	
		
		if style == 'fill_between':
			plt.fill_between(xm,yl,yh,interpolate=True,alpha=0.25,label=label,color=color)
		elif style == 'errorbar':
			#plt.errorbar(xm,ym,yerr=[ym-yl,yh-ym],fmt=color+'-o',label=label,lw=lw) 
			#plt.errorbar(xm,ym,yerr=[ym-yl,yh-ym],fmt=color,label=label,lw=lw)  #restore!! #+'-*'
			plt.errorbar(xm,ym,yerr=[abs(ym-yl),abs(yh-ym)],fmt=color,label=label,lw=lw)  #restore!! #+'-*'
			#plt.errorbar(xm,ym,yerr=[0.5*(yh-yl),0.5*(yh-yl)],fmt=color+'-*',label=label,lw=lw) 
	
		elif style == 'lines':
			if color == '':
				color=plt.gca()._get_lines.get_next_color()
			plt.plot(xm,ym,fmt,label=label,color=color,lw=lw,markersize=markersize)
			plt.plot(xm,yl,'--',color=color,lw=lw)
			plt.plot(xm,yh,'--',color=color,lw=lw)
		else: 
			if fig:				#added by Shourya (August 9, 2018)
				#plt.plot(xm,ym,color+fmt,label=label,lw=lw)
				plt.plot(xm,ym,fmt,label=label,color=color,lw=lw,markersize=markersize)
				# plt.plot(xm,ym,color+'-',label=label,lw=lw)
	
	ncount= []	
	indset = []
	for i in np.arange(h.bins):		
		ncount.append(h.indices(i).size)
		indset.append((h.indices(i)))
	ncount = np.array(ncount, dtype=object)
	indset = np.array(indset, dtype=object)
	


	if return_profile and return_bins:		
		return xm,ym,ncount[ncount>=mincount],indset, 0.5*(abs(ym-yl) + abs(yh-ym))
	elif return_profile:		
		return xm,ym






# SF stuff
# # global dthist_usehier
def comp_map_prl_exec(ival,typ='global'):
	'''
	INPUT: ival [pixel index from a Nd histogram]
	
	Purpose: Uses qnt_vec [vector of l,b,g,g_rp], to compute the completeness there
			 Mainly used for parallel implementation of completeness maps
			 
			 
			 
	'''
	if typ == 'global':
		use_sf = DR3SelectionFunctionTCG()
	elif typ == 'sub':
		use_sf = dtools.pickleread(sfdir+'/use_sf.pkl')
	qnt_vec = dtools.pickleread(sfdir+'/qntvec.pkl')

	l,b,g,g_rp = qnt_vec['l'][ival],qnt_vec['b'][ival],qnt_vec['phot_g_mean_mag'][ival],qnt_vec['G-Rp'][ival]
	l = np.array([l])
	b = np.array([b])
	val,unc_val = gunlim_comp(l,b,g,use_sf,grp=g_rp,typ=typ)
	comp_ = {}
	comp_['inum'] = np.array([ival])	
	comp_['compl'] = np.array([val])	
	comp_['unc_compl'] = np.array([unc_val])	
			
	
	return comp_
				
def make_compl_map_parallel(step_=2000,nchunks = 250,ncpu=28,savprefix='',typ='global'):
	
	
	'''
	requires: 
	
	desktop/
		temp_indres
		
		
		qntvec
		sf
	'''	
	
	from multiprocessing import Pool
	from functools import partial 
	indres = dtools.pickleread(sfdir+'/temp_indres.pkl')
	
	
	os.system('rm -rf '+desktop+'/runtime')
	os.system('mkdir '+desktop+'/runtime')

	inum = 0 	
	maxchunks_,remainstars_ = maxchunks(indres.size,step_)			

	timeit.begin()

	while inum < nchunks and inum <= maxchunks_  :			

		if inum < maxchunks_:

			inddelta = step_+(inum*step_)
			indx = indres[inum*step_:inddelta]
			nall = indx.size
		elif remainstars_ > 0:
			
			indx = indres[-remainstars_:]
			nall = indx.size				
		
		print('inum = '+str(inum))
		
		p=Pool(ncpu)
		# comp_res_set = p.map(comp_map_prl_exec,indx)
		comp_res_set = p.map(partial(comp_map_prl_exec, typ=typ), indx)	
		p.close()			
		timeit.end()
		
		comp_res={}
		for key in comp_res_set[0].keys():			
			comp_res[key]=np.concatenate([d[key] for d in comp_res_set]).transpose()		
		
		print('results obtained!..')
		dtools.ebfwrite(comp_res,'comp_res_'+str(inum),desktop+'/runtime')


		inum+= 1


	dtools.cmb_chunks(loc=desktop+'/runtime',svloc=sfdir,refkey='inum',read_format='.ebf',svname='compl_fac_'+savprefix+'_'+typ)

class rcselfunc(object):

	'''
	gaiaunlimited completeness grid stuff
	
	May 16, 2023 [INAF-TORINO]
	
	
	# # # plt.close('all')
	# # # gu = rcselfunc(magsuff='g')	
	# # # use_sf = 'gaia_wise_parallax'; savprefix='gaiawise'
	
	# # # dr3SubsampleSF = gu.gunlim_subsf(use_=use_sf,plot=True,plotmeth2=True,magmin=12,magmax=15,clruse=1.5,nocol=False)
	
	

	'''	
	def __init__(self,data=None,writehere='',magsuff='g',dlen=1000):
		
		'''
		magsuff = 'g' or 'w1' [the magnitude used to compute distances]
		'''
		
	
		
		if data == None:
			keepkys = ['pxgc','pygc','pzgc']			
			data = {}
			for ky in keepkys:
				data[ky+magsuff] = np.zeros(dlen)
			# raise RuntimeError('no data object provided')
			
		self.magsuff = magsuff
		data['pxgc'] = data['pxgc'+magsuff]
		data['pygc'] = data['pygc'+magsuff]
		data['pzgc'] = data['pzgc'+magsuff]


		self.data = data
		
		
		if 'l0' in data.keys():
			self.kymap = {'l':'l0','b':'b0','phot_g_mean_mag':'g','G-Rp':'g_rp'}
		else:
			self.kymap = {'l':'l','b':'b','phot_g_mean_mag':'g','G-Rp':'g_rp'}

		
		
		qfileloc = str(fetch_utils.get_datadir())+'/gfiles/qfiles'
		self.qfileloc = qfileloc		
		

	def grid2d_(self,			
			xmin=-10.,
			xmax = 10.,
			ymin=-10.,
			ymax = 10.,
			zmin=-10.,
			zmax = 10.,
			hbins = None,			
			hbinsx = 10,			
			hbinsy = 10,			
			hbinsz = 10,			
			normed=False,
			xy=False,xz=False,yz=False
			):


		val = locals()		
		self.args={}
		for ky in val.keys():			
			if ky !='self':
				self.args[ky] = val[ky]		
				
	
		ddum = self.data.copy()		
		

		h1 = data_hist2d_(ddum.copy(),xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,zmin=zmin,zmax=zmax,hbins=hbins,hbinsx=hbinsx,hbinsy=hbinsy,hbinsz=hbinsz,xy=xy,xz=xz,yz=yz,normed=normed) 	
	
		return h1

	def grid3d_(self,			
			xmin=-10.,
			xmax = 10.,
			ymin=-10.,
			ymax = 10.,
			zmin=-10.,
			zmax = 10.,
			hbins = None,			
			hbinsx = 100,			
			hbinsy = 100,			
			hbinsz = 100,			
			):


		val = locals()		
		self.args={}
		for ky in val.keys():			
			if ky !='self':
				self.args[ky] = val[ky]		
				
	
		ddum = self.data.copy()		
		
		h1 = data_hist3d_(ddum.copy(),xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,zmin=zmin,zmax=zmax,hbins=hbins,hbinsx=hbinsx,hbinsy=hbinsy,hbinsz=hbinsz) 	
	
				
		return h1


	def ncheck(self,h1):
		
		'''
		1. Saves qnt_vec [vector of l,b,g,g_rp] to sfdir
		2. Saves pixel indices to desktop with finite number of sources (to be run in paralell), to sfdir
		'''

		func=np.median		
		qnt_vec = {}		
		for ky in self.kymap:	
			print('running ..'+ky)		
			val = h1.apply(self.data[self.kymap[ky]],func)		
			qnt_vec[ky] = val.copy()
		
		qnt_vec['ncounts'] = h1.apply(self.data[self.kymap[ky]],len)
			
		self.qnt_vec = qnt_vec
		dtools.picklewrite(gu.qnt_vec,'qntvec',sfdir)
		
		timeit.end()
		# grab finite indices
		ky1 = 'G-Rp'
		ky2 = 'l'
		cond1 = np.isfinite((self.qnt_vec[ky1]))
		cond2 = np.isfinite((self.qnt_vec[ky2]))		
		self.indf = np.where(cond1&cond2)[0]
		timeit.end()		


		indres = (self.indf).copy()
		np.random.shuffle(indres)
		timeit.end()
				
		self.indres = indres
		dtools.picklewrite(self.indres,'temp_indres',sfdir)	


		return 


	def gunlim_queries(self):
		
		'''
		keeps a record of the queries run on the archive to obtain statistics to pass on to SF code
		
	

		Gaia X xp : gaia_xp.fits (37 minutes)

		SELECT healpix_, phot_g_mean_mag_, g_rp_, COUNT(*) AS n, SUM(selection) AS k
		                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 3)/1.)) AS phot_g_mean_mag_, to_integer(floor((g_rp - -1.0)/0.5)) AS g_rp_, to_integer(IF_THEN_ELSE(has_xp_continuous='true',1.0,0.0) ) AS selection
		                                        FROM gaiadr3.gaia_source  as gdr3            										 										 
												 WHERE phot_g_mean_mag > 3 AND phot_g_mean_mag < 20 AND g_rp > -1 AND g_rp < 3.0) AS subquery
		                                    GROUP BY healpix_, phot_g_mean_mag_, g_rp_		

		Gaia X xp [Wise] no clr : eloisa_query.fits (1 hour 10 minutes) - EP paper

		SELECT healpix_, phot_g_mean_mag_, COUNT(*) AS n, SUM(selection) AS k
		                                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 4)/1.)) AS phot_g_mean_mag_, to_integer(IF_THEN_ELSE(has_xp_continuous='true',1.0,0.0) ) AS selection
		                                                        FROM gaiadr3.gaia_source  as gdr3
		                                                                                                 left outer join gaiadr3.allwise_best_neighbour as wisenb USING (source_id)
		                                                                                                 WHERE phot_g_mean_mag > 4 AND phot_g_mean_mag < 15 and wisenb.source_id is not null ) AS subquery
		                                                    GROUP BY healpix_, phot_g_mean_mag_		
		
		
		
		Gaia X xp [Wise] : gaia_xp_wise.fits (1 hour 10 minutes)
		SELECT healpix_, phot_g_mean_mag_, g_rp_, COUNT(*) AS n, SUM(selection) AS k
		                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 3)/1.)) AS phot_g_mean_mag_, to_integer(floor((g_rp - -1.0)/0.5)) AS g_rp_, to_integer(IF_THEN_ELSE(has_xp_continuous='true',1.0,0.0) ) AS selection
		                                        FROM gaiadr3.gaia_source  as gdr3            										 										 
												 left outer join gaiadr3.allwise_best_neighbour as wisenb USING (source_id) 		
												 WHERE phot_g_mean_mag > 3 AND phot_g_mean_mag < 20 AND g_rp > -1 AND g_rp < 3.0 and wisenb.source_id is not null ) AS subquery
		                                    GROUP BY healpix_, phot_g_mean_mag_, g_rp_
                             


		Gaia X XP [wise, rc selection using source ids]: gaia_xp_wise_rc.fits (41 minutes)
		SELECT healpix_, phot_g_mean_mag_, g_rp_, COUNT(*) AS n, SUM(selection) AS k
				                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 3)/1.)) AS phot_g_mean_mag_, to_integer(floor((g_rp - -1.0)/0.5)) AS g_rp_, to_integer(IF_THEN_ELSE('source_id in (select source_id from user_skhann01.table1)' ,1.0,0.0) 
																																																													) AS selection
				       FROM gaiadr3.gaia_source  as gdr3            										 										 
														
														 WHERE phot_g_mean_mag > 3 AND phot_g_mean_mag < 20 AND g_rp > -1 AND g_rp < 3.0) AS subquery
				                                    GROUP BY healpix_, phot_g_mean_mag_, g_rp_	
		


		Gaia X XP [catwise, rc selection using source ids]: gaia_xp_catwise_rc.fits (41 minutes)
		SELECT healpix_, phot_g_mean_mag_, g_rp_, COUNT(*) AS n, SUM(selection) AS k
				                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 3)/1.)) AS phot_g_mean_mag_, to_integer(floor((g_rp - -1.0)/0.5)) AS g_rp_, to_integer(IF_THEN_ELSE('source_id in (select source_id from user_skhann01.table1)' ,1.0,0.0) 
																																																													) AS selection
				       FROM gaiadr3.gaia_source  as gdr3            										 										 
														
														 WHERE phot_g_mean_mag > 3 AND phot_g_mean_mag < 20 AND g_rp > -1 AND g_rp < 3.0) AS subquery
				                                    GROUP BY healpix_, phot_g_mean_mag_, g_rp_	
		



		
		Gaia X Rave  : gaia_rave.fits (37 minutes)		
		SELECT healpix_, phot_g_mean_mag_, g_rp_, COUNT(*) AS n, SUM(selection) AS k
		                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 3)/1.)) AS phot_g_mean_mag_, to_integer(floor((g_rp - -1.0)/0.5)) AS g_rp_, to_integer(IF_THEN_ELSE(ravenb.source_id >0,1.0,0.0)) AS selection
		                                        FROM gaiadr3.gaia_source  as gdr3            
												 left outer join gaiadr3.ravedr6_best_neighbour as ravenb USING (source_id)		
												 WHERE phot_g_mean_mag > 3 and phot_g_mean_mag < 16 and g_rp > -1 and g_rp < 5) AS subquery
		                                    GROUP BY healpix_, phot_g_mean_mag_, g_rp_			

		
		
		Gaia X WISE : gaia_wise.fits ( minutes)
		
		SELECT healpix_, phot_g_mean_mag_, g_rp_, COUNT(*) AS n, SUM(selection) AS k
		                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 3)/1.)) AS phot_g_mean_mag_, to_integer(floor((g_rp - -1.0)/0.5)) AS g_rp_, to_integer(IF_THEN_ELSE(wisenb.source_id >0,1.0,0.0) ) AS selection
		                                        FROM gaiadr3.gaia_source  as gdr3            										 										 
												 left outer join gaiadr3.allwise_best_neighbour as wisenb USING (source_id) 		
												 WHERE phot_g_mean_mag > 3 AND phot_g_mean_mag < 20 AND g_rp > -1 AND g_rp < 3.0) AS subquery
		                                    GROUP BY healpix_, phot_g_mean_mag_, g_rp_			
		
		
		Gaia X WISE [parallax]: gaia_wise_parallax.fits ( minutes)	
		
		SELECT healpix_, phot_g_mean_mag_, g_rp_, COUNT(*) AS n, SUM(selection) AS k
		                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 3)/1.)) AS phot_g_mean_mag_, to_integer(floor((g_rp - -1.0)/0.5)) AS g_rp_, to_integer(IF_THEN_ELSE(wisenb.source_id >0,1.0,0.0) ) AS selection
		                                        FROM gaiadr3.gaia_source  as gdr3            										 										 
												 left outer join gaiadr3.allwise_best_neighbour as wisenb USING (source_id) 		
												 WHERE phot_g_mean_mag > 3 AND phot_g_mean_mag < 20 AND g_rp > -1 AND g_rp < 3.0 and parallax is not null) AS subquery
		                                    GROUP BY healpix_, phot_g_mean_mag_, g_rp_	


		Gaia X WISE [parallax]_noclr: gaia_wise_parallax_noclr.fits ( minutes)	
		
		SELECT healpix_, phot_g_mean_mag_, COUNT(*) AS n, SUM(selection) AS k
		                                    FROM (SELECT to_integer(GAIA_HEALPIX_INDEX(5,source_id)) AS healpix_, to_integer(floor((phot_g_mean_mag - 3)/1.)) AS phot_g_mean_mag_,to_integer(IF_THEN_ELSE(wisenb.source_id >0,1.0,0.0) ) AS selection
		                                        FROM gaiadr3.gaia_source  as gdr3            										 										 
												 left outer join gaiadr3.allwise_best_neighbour as wisenb USING (source_id) 		
												 WHERE phot_g_mean_mag > 3 AND phot_g_mean_mag < 20 AND g_rp > -1 AND g_rp < 3.0 and parallax is not null) AS subquery
		                                    GROUP BY healpix_, phot_g_mean_mag_	



				
		'''				
						 
		return 


	def gunlim_subsf(self,use_=None,plot=False,plotmeth2=False,magmin=10,magmax=15,clruse = 0.5,nocol=False):

		'''
		This grabs the selection function for a specific pre-run query
		
		use_ = None
		use_ = gaia_xp_allwise_plx 
		use_ =  
		'''
	
		if use_ == None:
			raise RuntimeError('specify query time to use \ gaia_xp_allwise_plx \ ')			
			
		
		if use_ == 'gaia_xp': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1] , 'g_rp': [-1 ,3 ,0.5]}
			query_filename = use_ ; ttl = 'Gaia X xp '
		if use_ == 'gaia_xp_wise': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1] , 'g_rp': [-1 ,3 ,0.5]}
			query_filename = use_ ; ttl = 'Gaia X xp X wise '
		if use_ == 'gaia_wise_plx': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1] , 'g_rp': [-1 ,3 ,0.5]}
			query_filename = use_ ; ttl = 'Gaia X wise X plx '
		if use_ == 'gaia_wise': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1] , 'g_rp': [-1 ,3 ,0.5]}
			query_filename = use_ ; ttl = 'Gaia X wise  '
		if use_ == 'gaia_wise_parallax': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1] , 'g_rp': [-1 ,3 ,0.5]}
			query_filename = use_ ; ttl = 'Gaia X wise X parallax clr'
		if use_ == 'gaia_wise_parallax_noclr': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1]}
			query_filename = use_ ; ttl = 'Gaia X wise X parallax '


		if use_ == 'eloisa_query':
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [4 ,15 ,1]}
			query_filename = use_ ; ttl = 'Gaia XP x WISE '
		if use_ == 'gaia_xp_wise_rc':
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1] , 'g_rp': [-1 ,3 ,0.5]}
			query_filename = use_ ; ttl = 'Gaia X xp X wise (rc) '
			
		if use_ == 'gaia_xp_catwise_rc':
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1] , 'g_rp': [-1 ,3 ,0.5]}
			query_filename = use_ ; ttl = 'Gaia X xp X catwise (rc) '
			


		if use_ == 'rvssubsel': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,0.2] , 'g_rp': [-2.5 ,5.1 ,0.4]}
			query_filename = use_  ; ttl = 'rvs '
			
		if use_ == 'rvssubsel_widebins': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,20 ,1.0] , 'g_rp': [-1.0 ,3. ,0.5]}
			query_filename = use_  ; ttl = 'rvs '
		
		if use_ == 'gaia_rave': 
			
			hplevel_and_binning = {'healpix': 5, 'phot_g_mean_mag': [3 ,16 ,1.] , 'g_rp': [-1.0 ,5. ,0.5]}
			query_filename = use_  ; ttl = ' '



		qfileloc = self.qfileloc
		file_name = query_filename+'_modif'		
		print(file_name)
		mydata = tabpy.read(qfileloc+'/'+query_filename+'.fits')
		t = Table(mydata)
		self.t = t		

		dr3SubsampleSF =  sksubsamplesf(mydata,query_filename,file_name,hplevel_and_binning)
		pixtomagfac_1 = hplevel_and_binning['phot_g_mean_mag'][2]
		pixtomagfac_2 = hplevel_and_binning['phot_g_mean_mag'][0]

		
		magindxs = np.unique(np.array(t['phot_g_mean_mag_']))
		mags = (magindxs*pixtomagfac_1)+pixtomagfac_2		
		indmags = np.where((mags[magindxs] > magmin)&(mags[magindxs] < magmax))[0]		

		if 'g_rp' in hplevel_and_binning.keys():
			pixtoclrfac_1 = hplevel_and_binning['g_rp'][2]
			pixtoclrfac_2 = hplevel_and_binning['g_rp'][0]
	
	
	

			
			
			clrindxs = (np.unique(np.array(t['g_rp_'])))
			clrs = (clrindxs*pixtoclrfac_1) + pixtoclrfac_2
			
			from scipy import interpolate
			f = interpolate.interp1d(clrs,clrindxs,bounds_error=False,fill_value=(clrindxs[0],clrindxs[-1]))
			clrbin = f(clruse)
			
			clrmin = (clrbin*pixtoclrfac_1) + pixtoclrfac_2
			clrmax = clrmin + pixtoclrfac_1


		if plot:
			plt.close('all')		

			
			t['hpx4'] = t['healpix_'] // 4
			t['hpx3'] = t['healpix_'] // 16
			t['hpx2'] = t['healpix_'] // 64
			t['hpx1'] = t['healpix_'] // 256
			t['hpx0'] = t['healpix_'] // 1024			
			
			t_only_by_mag = t.group_by(['phot_g_mean_mag_']).groups.aggregate(np.sum)
			
			k_over_n = t_only_by_mag['k'] / t_only_by_mag['n']
			plt.step( (t_only_by_mag['phot_g_mean_mag_']*pixtomagfac_1)+pixtomagfac_2 , k_over_n , where='post')
			plt.minorticks_on()
			plt.xlabel('G')
			plt.ylabel('fraction of sources:'+query_filename)
			plt.ylim(-0.03,1.03)
			
			plt.savefig(desktop+'/rubfigs/magonly.png')		
				

						
			for magindx in magindxs[indmags]: #[50:56] 
			
				if nocol:
					t_colourMagnitudeBin = t[ (t['phot_g_mean_mag_']==magindx) ]
					clrmin = ''
					clrmax = ''
				else:
					t_colourMagnitudeBin = t[ (t['phot_g_mean_mag_']==magindx) & (t['g_rp_']==clrbin) ]

				plt.close('all')
				
				print(magindx,clrbin)
				print('at this magnitude ')
				print(str((magindx*pixtomagfac_1)+pixtomagfac_2))

				print(clrmin, clrmax)
				import healpy as hp
				
				plt.figure(figsize=(12,6))
				
				for lll,level in enumerate(['hpx0','hpx1','hpx2','hpx3','hpx4','healpix_']):
					t_lowres = t_colourMagnitudeBin.group_by([level]).groups.aggregate(np.sum)
					ratios_lowres = t_lowres['k']/t_lowres['n']
					hpxNb_lowres = t_lowres[level]
					
					# Some might be empty, so we make a new list with the correct length and fill it:
					ratios_lowres_nogaps = np.empty(12*4**lll)
					ratios_lowres_nogaps[:] = np.nan
					for iii in range(len(ratios_lowres_nogaps)):
						try:
							ratios_lowres_nogaps[iii] = ratios_lowres[ t_lowres[level]==iii ][0]
						except:
							pass #we have no counts in there
						
					plt.subplot(2,3,lll+1)
					hp.mollview( ratios_lowres_nogaps , nest=True , coord='CG',hold=True,min=0,max=1.,
							   title='healpix level %i' % (lll) )		
					
				plt.text(0.01,0.9,str(clrmin)+'_'+str(clrmax),transform=plt.gca().transAxes,fontsize=10,color='black',rotation=0)	
							   
				plt.savefig(desktop+'/rubfigs/rubbish'+str((magindx*pixtomagfac_1)+pixtomagfac_2)+'.png')       
			       	

		if plotmeth2:			
			
			
			import healpy as hp
			plt.close('all')
			# cmap=cc.cm['linear_green_5_95_c69'] 
			# cmap=cc.cm['linear_kbgoy_20_95_c57'] 
			cmap=cc.cm['bgy'] 
			
			# plm=putil.Plm1(3,3,xsize=8.0,ysize=8.,xmulti=False,ymulti=False,full=True,slabelx=0.7,slabely=0.07)
			# plm=putil.Plm1(1,1,xsize=8.0,ysize=8.,xmulti=False,ymulti=False,full=True,slabelx=0.7,slabely=0.07)
			

			healpix_level = 5
			nside = 2**healpix_level
			totpix = 12*(nside**2)	
			area_per_pix = dtools.allsky_area()/(totpix)
			print(' area per pixel is,')
			print(area_per_pix)
			
		

			
			coords_of_centers = get_healpix_centers (healpix_level)
			# for inum,gmags in enumerate(range(6,20,1)):
			for inum,gmags in enumerate(range(10,17,1)):
			# for inum,gmags in enumerate(range(10,14,1)):
							
				# plm.next()
				plt.close('all')
				print(gmags)			
				G = gmags
				G_RP = clruse

				gmag = np. ones_like(coords_of_centers)*G
				col = np. ones_like(coords_of_centers)*G_RP
				completeness,unc_completeness=dr3SubsampleSF.query(coords_of_centers,phot_g_mean_mag_=gmag,g_rp_=col)					
				

				hp.mollview( completeness , nest=False , coord='CG',hold=False,min=0,max=1.,
					   title=str(gmags)+'<G<'+str(gmags+1),cmap=cmap,notext=True,badcolor="white",unit="Completeness") #,norm="hist"			
				hp.graticule()	  
				
				# projview(completeness.values,coord=['C','G'],min=0,max=1,graticule=True,cbar=True,graticule_labels=True,xlabel="l",ylabel="b",projection_type="mollweide",cmap=cmap,cb_orientation="horizontal")		

				# projview(completeness.values,coord=["G"],min=0,max=1,graticule=True,cbar=True,graticule_labels=True,xlabel="l",ylabel="b",projection_type="aitoff",cmap=cmap,cb_orientation="horizontal")							   
				
				# hp.mollview(completeness.values,coord=['C','G'], hold=True,cbar=True,min=0,max=1,title=str(gmags),cmap=cmap,badcolor="white"whel

				# hp.mollview(completeness.values,coord=['C','G'], hold=True,cbar=True,min=0,max=1,title=str(gmags))#,cmap=cmap,badcolor="white")
			
			
			# plt.figtext(0.1,0.9,ttl,transform=plt.gca().transAxes,weight='bold',color='red')		
			# plm.tight_layout()
			
				plt.savefig(desktop+'/rubfigs/compsky_g'+str(gmags)+'_'+str(inum)+'.png')
		
		return dr3SubsampleSF


def compgusf_grid(l,
				b,
				g,
				grp=-10,
				typ_='global',
				use_subsf = 'gaia_wise_parallax_noclr'):
	
	'''
	calculates the completeness for given:
		
	l,b,g,grp=-10
	typ_ = 'global', 'sub'
	
	use_sf = 'gaia_wise_parallax_noclr'; savprefix='gaiawise'	
	
	HISTORY : 31 May 2024 (INAF-Torino)
	
	
	'''
		
	qfileloc = str(fetch_utils.get_datadir())+'/gfiles/qfiles'
	gu = rcselfunc(magsuff='g')	
	
	if typ_ == 'sub':
		sf = gu.gunlim_subsf(use_=use_subsf)
	if typ_ =='global':
		sf = DR3SelectionFunctionTCG()

		
	vals,unc_vals = gunlim_comp(l,b,g,sf,typ=typ_,grp=grp)
	
	# for anything brighter than G = 11, assume total completeness (as there are too few sources to get the ratio-ing to work correctly here) - added June 25, 2024
	indbr = np.where(g < 11)[0]
	vals[indbr] = 1.
	
	return vals,unc_vals


def binhierc_cylin3d(dthist_use2,h2,nbinsR=10,nbinsphi=10,nbinsz=10,Rmin=0,Rmax=20,phimin=0,phimax=360.,
					zmin=-2,zmax=2.,extinct_=False,add_dust_to_grid_2d=False,add_dust_to_grid_3d=False,binfac=5,mkplots_=False,compsf=True,gusf=True,dmaglim=17,
					use_subsf='gaia_wise_parallax_noclr'):
	
	
	'''
	hierarchical binning for cylindrical histogram to compute the sf due to maglim/extinction
	'''


	
	nbinsR3 = int(binfac*nbinsR)
	nbinsphi3 = int(binfac*nbinsphi)
	nbinsz3 = int(binfac*nbinsz)

	# nbinsR3 = int(binfac*nbinsR)
	# nbinsphi3 = int(1.*nbinsphi)
	# nbinsz3 = int(1.*nbinsz)
	
	
	xbnwd = (h2.range[0][1] - h2.range[0][0])/h2.bins[0]	
	ybnwd = (h2.range[1][1] - h2.range[1][0])/h2.bins[1]
	zbnwd = (h2.range[2][1] - h2.range[2][0])/h2.bins[2]	


	xedges_ind_min = dthist_use2['rgcv'] - xbnwd/2.
	xedges_ind_max = dthist_use2['rgcv'] + xbnwd/2.

	yedges_ind_min = dthist_use2['phiv'] - ybnwd/2.
	yedges_ind_max = dthist_use2['phiv'] + ybnwd/2.

	zedges_ind_min = dthist_use2['zv'] - zbnwd/2.
	zedges_ind_max = dthist_use2['zv'] + zbnwd/2.


	mydata_ = {}
	mydata_['rgc'] = np.linspace(0.,10,100)
	mydata_['phi1'] = np.linspace(0.,10,100)
	mydata_['pzgc'] = np.linspace(0.,10,100)
	
	plt.close('all')

	h3 = sutil.hist_nd([mydata_['rgc'],mydata_['phi1'],mydata_['pzgc']],bins=[nbinsR3,nbinsphi3,nbinsz3],normed=False,range=[[Rmin,Rmax],[phimin,phimax],[zmin,zmax]])

	dthist_use3 = cylingrid(h3,add_dust_to_grid_2d=add_dust_to_grid_2d,add_dust_to_grid_3d=add_dust_to_grid_3d)
	dthist_use3['gmag_pred'] = autil.dist2dmod(dthist_use3['dist']) + absmag_lit['absg']


	xbnwd3 = (h3.range[0][1] - h3.range[0][0])/h3.bins[0]
	ybnwd3 = (h3.range[1][1] - h3.range[1][0])/h3.bins[1]
	zbnwd3 = (h3.range[2][1] - h3.range[2][0])/h3.bins[2]

	xedges3_ind_min = dthist_use3['rgcv'] - xbnwd3/2.
	xedges3_ind_max = dthist_use3['rgcv'] + xbnwd3/2.

	yedges3_ind_min = dthist_use3['phiv'] - ybnwd3/2.
	yedges3_ind_max = dthist_use3['phiv'] + ybnwd3/2.	

	zedges3_ind_min = dthist_use3['zv'] - zbnwd3/2.
	zedges3_ind_max = dthist_use3['zv'] + zbnwd3/2.	



	tmpfile = {}
	tmpfile['xedges_ind_min'] = xedges_ind_min
	tmpfile['xedges_ind_max'] = xedges_ind_max
	tmpfile['yedges_ind_min'] = yedges_ind_min
	tmpfile['yedges_ind_max'] = yedges_ind_max
	tmpfile['zedges_ind_min'] = zedges_ind_min
	tmpfile['zedges_ind_max'] = zedges_ind_max

	tmpfile['xedges3_ind_min'] = xedges3_ind_min
	tmpfile['xedges3_ind_max'] = xedges3_ind_max
	tmpfile['yedges3_ind_min'] = yedges3_ind_min
	tmpfile['yedges3_ind_max'] = yedges3_ind_max
	tmpfile['zedges3_ind_min'] = zedges3_ind_min
	tmpfile['zedges3_ind_max'] = zedges3_ind_max
	tmpfile['dmaglim'] = dmaglim
	tmpfile['extinct_'] = extinct_
	

	dtools.picklewrite(tmpfile,'tmpfile',tempdir)
	dtools.picklewrite(dthist_use3,'dthisttmp3',tempdir)


	timeit.end()


	if compsf:

		indres = np.arange(0,xedges_ind_min.size)
		from multiprocessing import Pool
		p=Pool(30)
		data_postdist = p.map(partial(hiercalc_cylin3d), indres)
		data_pd={}
		for key in data_postdist[0].keys():			
			data_pd[key]=np.concatenate([d[key] for d in data_postdist]).transpose()	
					
								
	
		p.close()
	

	timeit.end()
	



	if gusf:
			
		data_pd['subsf'],data_pd['unc_subsf'] = compgusf_grid(dthist_use3['l'],dthist_use3['b'],dthist_use3['gmag_pred'],typ_='sub',use_subsf=use_subsf)
		data_pd['globsf'],data_pd['unc_globsf'] = compgusf_grid(dthist_use3['l'],dthist_use3['b'],dthist_use3['gmag_pred'],typ_='global',use_subsf=use_subsf)	



	dtools.picklewrite(data_pd,'sfhier_interm',sfdir)


	if gusf:

		sftyp = 'subsf'
		
		indres = np.arange(0,xedges_ind_min.size)
		from multiprocessing import Pool
		p=Pool(30)
		data_postdist = p.map(partial(hiercalc_cylin3d_gusf,typ_=sftyp), indres)
		data_pd1={}
		for key in data_postdist[0].keys():			
			data_pd1[key]=np.concatenate([d[key] for d in data_postdist]).transpose()	
							
		
		p.close()	
		
		data_pd['subsf_avg'] = data_pd1['sfval'].copy()
		data_pd['subsf_std'] = data_pd1['sfval_std'].copy()
			
		timeit.end()	
		
		sftyp = 'globsf'
		
		indres = np.arange(0,xedges_ind_min.size)
		from multiprocessing import Pool
		p=Pool(30)
		data_postdist = p.map(partial(hiercalc_cylin3d_gusf,typ_=sftyp), indres)
		data_pd1={}
		for key in data_postdist[0].keys():			
			data_pd1[key]=np.concatenate([d[key] for d in data_postdist]).transpose()	
							
		
		p.close()		

		data_pd['globsf_avg'] = data_pd1['sfval'].copy()
		data_pd['globsf_std'] = data_pd1['sfval_std'].copy()

		timeit.end()		
	
	
	if gusf == False:

		data_pd['globsf_avg'] = np.zeros(data_pd['sf'].size) + 1.
		data_pd['globsf_std'] = np.zeros(data_pd['sf'].size) + 1.

		data_pd['subsf_avg'] = np.zeros(data_pd['sf'].size) + 1.
		data_pd['subsf_std'] = np.zeros(data_pd['sf'].size) + 1.
			
	
	return data_pd
	
	
	
def hiercalc_cylin3d_gusf(i,typ_='sub'):
	
	'''
	typ_ = 'sub' or 'global'

	'''
	
	tmpfile = dtools.pickleread(tempdir+'/tmpfile.pkl')
	sfcomp = dtools.pickleread(sfdir+'/sfhier_interm.pkl')	
	
	xedges_ind_min = tmpfile['xedges_ind_min']
	xedges_ind_max = tmpfile['xedges_ind_max']
	yedges_ind_min = tmpfile['yedges_ind_min'] 
	yedges_ind_max = tmpfile['yedges_ind_max']
	zedges_ind_min = tmpfile['zedges_ind_min'] 
	zedges_ind_max = tmpfile['zedges_ind_max'] 
	
	xedges3_ind_min = tmpfile['xedges3_ind_min'] 
	xedges3_ind_max = tmpfile['xedges3_ind_max']
	yedges3_ind_min = tmpfile['yedges3_ind_min']
	yedges3_ind_max = tmpfile['yedges3_ind_max']
	zedges3_ind_min = tmpfile['zedges3_ind_min']
	zedges3_ind_max = tmpfile['zedges3_ind_max']
	dmaglim = tmpfile['dmaglim']
	extinct_ = tmpfile['extinct_']
		

	cond1 = (xedges3_ind_min >= xedges_ind_min[i])&(xedges3_ind_min < xedges_ind_max[i])	
	cond2 = (yedges3_ind_min >= yedges_ind_min[i])&(yedges3_ind_max <= yedges_ind_max[i])
	cond3 = (zedges3_ind_min >= zedges_ind_min[i])&(zedges3_ind_max <= zedges_ind_max[i])
	conds = cond1&cond2&cond3
	indf = np.where(conds)[0]
	


	tmpval = sfcomp[typ_][indf]

	res = {}
	res['sfval'] = np.array([np.median(tmpval)])
	res['sfval_std'] = np.array([np.std(tmpval)])	

	return res

	
def hiercalc_cylin3d(i):
	
	'''
	dthist_usehier is set to global
	'''
	
	# print(i)
	tmpfile = dtools.pickleread(tempdir+'/tmpfile.pkl')
	
	dthist_usemain = dtools.pickleread(tempdir+'/dthist1_rphiz.pkl')
	dthist_use3 = dtools.pickleread(tempdir+'/dthisttmp3.pkl')
	
	
	xedges_ind_min = tmpfile['xedges_ind_min']
	xedges_ind_max = tmpfile['xedges_ind_max']
	yedges_ind_min = tmpfile['yedges_ind_min'] 
	yedges_ind_max = tmpfile['yedges_ind_max']
	zedges_ind_min = tmpfile['zedges_ind_min'] 
	zedges_ind_max = tmpfile['zedges_ind_max'] 
	
	xedges3_ind_min = tmpfile['xedges3_ind_min'] 
	xedges3_ind_max = tmpfile['xedges3_ind_max']
	yedges3_ind_min = tmpfile['yedges3_ind_min']
	yedges3_ind_max = tmpfile['yedges3_ind_max']
	zedges3_ind_min = tmpfile['zedges3_ind_min']
	zedges3_ind_max = tmpfile['zedges3_ind_max']
	dmaglim = tmpfile['dmaglim']
	extinct_ = tmpfile['extinct_']
		

	cond1 = (xedges3_ind_min >= xedges_ind_min[i])&(xedges3_ind_min < xedges_ind_max[i])	
	cond2 = (yedges3_ind_min >= yedges_ind_min[i])&(yedges3_ind_max <= yedges_ind_max[i])
	cond3 = (zedges3_ind_min >= zedges_ind_min[i])&(zedges3_ind_max <= zedges_ind_max[i])
	conds = cond1&cond2&cond3
	indf = np.where(conds)[0]
	
		
	
	if extinct_:

		dmaxs = autil.dmod2dist(dmaglim - dthist_use3['a_g_val'][indf] - absmag_lit['absg'])		
		dmaxs_main = autil.dmod2dist(dmaglim - dthist_usemain['a_g_val'][i] - absmag_lit['absg'])	
		
		
	else:
				
		dmaxs = autil.dmod2dist((dmaglim) - absmag_lit['absg'])	
		dmaxs_main = dmaxs

	# # #------------checking
	# # tst = {}
	# # tst['indf'] = indf
	# # tst['i'] = i
	# # dtools.picklewrite(tst,'rubbish',desktop,prnt=False)
	# # #--------------------
	
	# print('check....')
	# print(i,indf)
	# print(dmaxs.size)
	# print(dmaxs)


	indgmax = np.where(dthist_use3['dist'][indf] < dmaxs)[0]
	
	sfval = indgmax.size/indf.size

	
	if extinct_:

		gvals = autil.dist2dmod(dthist_use3['dist'][indf]) + absmag_lit['absg'] +dthist_use3['a_g_val'][indf]
		gmax = np.nanmax(gvals)
		gmin = np.nanmin(gvals)
		gmed = np.median(gvals)
	
	if dthist_usehier['dist'][i] < dmaxs_main :
		sfval_wrong = 1

	else:
		sfval_wrong = 0

	
	res = {}
	res['sf'] = np.array([sfval])
	res['sfval_wrong'] = np.array([sfval_wrong])
	res['gmax'] = np.array([gmax])
	res['gmin'] = np.array([gmin])
	res['gmed'] = np.array([gmed])
	return res

