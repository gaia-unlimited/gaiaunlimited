import os, sys, time, math
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


# import agama
# import vaex
# matplotlib.use('Agg') # non interactive
# matplotlib.use('Qt5Agg')

# bmcmc = dtools.get_bmcmc2()




##### utility functions

																			
#---Toolssc

# def getdirec(loc,at='mac'):
# def getdirec(loc,at='bologna'):
def getdirec(loc,at='cluster2'):
# def getdirec(loc,at='torino'):
	'''
	OPTIONS:
	1. home: '/home/shourya' 
	2. phddir: home+'/Documents/phd_work'
	3. glxdir: phddir+'/GalaxiaWork'
	3. desktop: home+'/Desktop'
	4. downloads: home+'/Downloads'
	5. galaxia: home+'/GalaxiaData'

	at = ['kapteyn','lenovo','mac','INAF','cluster','cluster2','torino']

	'''
	#savdir = HOME+'/warp2018'

	# home= '/home/shourya';	galaxia = home+'/GalaxiaData'	
	# home= '/import/photon1/skha2680';galaxia = '/import/photon1/sanjib/share/GsynthData'

	if at == 'kapteyn':
		home= '/net/huygens/data/users/khanna';
	if at == 'mac':
		home= '/Users/shouryapro';
	if at == 'bologna':
		home= '/home/guest/Sunny/data8';
	if at == 'cluster':
		home= '/u/skhanna';
	if at == 'cluster2':
		home= '/iranet/users/pleia14/';
	if at == 'torino':
		home= '/home/shouryakhanna';
	

	desktop = home+'/Desktop'
	documents = home+'/Documents'
	downloads = home+'/Downloads'
	pdocdir= home+'/Documents/pdoc_work'  
	galaxia = documents+'/GalaxiaData'	
	phddir= home+'/Documents/phd_work'  	
	glxdir = phddir+'/GalaxiaWork'

	gaiadata = home+'/Documents/phd_work/gaia/data'
	
	direc = {'home':home,'phddir':phddir,'pdocdir':pdocdir,'glxdir':glxdir,'desktop':desktop,'downloads':downloads,'galaxia':galaxia,'gaiadata':gaiadata,'documents':documents}
	
	return direc[loc]
			
	
def mk_mini(mydata,fac=1.0,key='pz',prnt_=True):		
	
	import tabpy 

	putvelkeysback = False
	#velkeys = ['usun','vsun','wsun','xsun','zsun']
	
	velkeys = []
	for ky in mydata.keys():
		if mydata[ky].size < mydata[key].size:
			velkeys.append(ky)
	velkeys = np.array(velkeys)
	tmp_vel={}
	
	for ky in velkeys:						
		if ky in mydata.keys():						
			tmp_vel[ky] = mydata[ky]		
			del mydata[ky]
			putvelkeysback = True

	
	if prnt_:
		print(mydata[key].size,' stars in all')		



	crnt = (mydata[key].size)*fac; crnt = int(crnt)	
	tabpy.shuffle(mydata,crnt)
	if prnt_:
		print(mydata[key].size,' stars shuffled')	
	
	
	if putvelkeysback:				
		for ky in velkeys:				
				mydata[ky] = tmp_vel[ky]
		
	return mydata
def filter_dict(data,ind):	
	'''
	PURPOSE
	
	> Filters given dictionary by given indices
	  equivalent of doing mydata['x'] =  mydata['x'][ind] for each key
	
	INPUT:
	> data: dictionary containing some data
	> ind: indices at which to filter data
	!!! all keys need to have same length !!!
	
	RETURNS:
	> data filtered at specified indices	
	'''
	
	data1={}
	for key in data.keys():				
		# print(key)
		data1[key]=data[key][ind]
	return data1


def indfinite(qnt,nans=False,print_=True):
	'''
	Returns the indices where input quantity is finite
	
	'''
	if nans == False:
		ind = np.where(np.isfinite(qnt))[0]
		if print_:
			print('ind where finite')
			print(ind.size)		
	else:

		ind = np.where(np.isfinite(qnt)==False)[0]		
		if print_:		
			print('ind where not finite')
			print(ind.size)	
	return ind


def sqrtsum(ds=[],prnt=False):
	'''
	handy function to sum up the squares and return the square-root
	'''
	
	if prnt:
		print(len(ds))
	
	mysum = 0
	for i in range(len(ds)):
		
		
		tmp = ds[i]**2.
		mysum+=tmp
	
	
	resval = np.sqrt(mysum)
	
	
	return resval



def add_array_nan(im1,im2):
	
	import numpy
		
	# a = im1.copy()
	# b = im2.copy()       
	na = numpy.isnan(im1)
	nb = numpy.isnan(im2)
	im1[na] = 0
	im2[nb] = 0
	im1 += im2
	na &= nb
	im1[na] = numpy.nan	
	

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




# kinematics
def get_lsr(typ='schonrich',show=False):
	'''
	Comment: using 8.275 kpc following ESO-grav 2021
			 # was using 8.2 kpc following ESO-grav 2018
			 
	typ =='schonrich':
	from schonrich 2011
	 
		
		
	typ == 'gravity':	       
	# values used in Drimmel 2022 (based on Gravity)  
	
	
	typ == 'galaxia':
	for use with galaxia kinematics		
		
	'''
	
	# print(typ)
	
	LSR = {}
	if typ =='schonrich':
		LSR = {'xsun':-8.275,
		       'zsun':0.027,
		       'usun':11.1,
		       'wsun':7.25,
		       'omega_sun':30.24}
	elif typ == 'gravity':	       
		# values used in Drimmel 2022 (based on Gravity)       
		LSR = {'xsun':-8.277,
		       'zsun':0.,
		       'usun':9.3,
		       # 'omega_sun':6.39*4.74}
		       'omega_sun':6.411*4.74}
     
		LSR['wsun'] = 0.219*4.74*LSR['xsun']*-1
     		       

	elif typ == 'galaxia':
		LSR = {'xsun':-8.,
		       'zsun':0.0,
		       'usun':11.1,
		       'wsun':7.25,
		       'omega_sun':239.08/8.}
				       
	LSR['vsun'] = -LSR['omega_sun']*LSR['xsun']
	LSR['Rsun'] = -LSR['xsun']
	
	if show:

		print('')
		print(typ+' LSR: ')	
		for k, v in LSR.iteritems():		
			print ("{:<15} :{:<15}".format(k,v))
	
	return LSR
	
	
def getkin(d,rkey,rad_vel_key='vr',typ='schonrich'):	
	
	'''
	
	input: 
	d = dictionary of data - as a minimum requiring the columns [d['ra'],d['dec'],d[rkey],d['mura'],d['mudec'],d[rad_vel_key]]
	where mura = pmra*1e-03
	rkey = the key in the dictionary representing heliocentric distance in kpc
	rad_vel_key = the key in the dictionary representing line-of-sight velocity in km/s
	typ = which LSR constants to use ['schonrich','gravity','galaxia'] - this calls get_lsr(typ)
	
	Output:
	R (galactocentric) = 'rgc'
	r (galactocentric) = 'rgcd'
	l = glon
	b = glat
	
	'''
	
	print('')
	print('rkey = '+rkey)
	lsr = get_lsr(typ=typ)	
	print(lsr)
	
	omega_sun = lsr['omega_sun'];usun=lsr['usun'];wsun=lsr['wsun'];xsun=lsr['xsun'];zsun=lsr['zsun']				
	
	vsun= -(omega_sun*xsun)	
	
	
	# print('')
	# print('Using '+dtype+' kinematics')
		
			
	# print('')
	# print('Using '+rad_vel_key)	
		
																																																																												
	# print(xsun)

	if 'l' in d.keys():
		d['glon'] = d['l'].copy()
		d['glat'] = d['b'].copy()

	if 'glon' not in d.keys() and 'l' not in d.keys():
		d['glon'],d['glat']=autil.equ2gal(d['ra'],d['dec'])
	
	
		
	if 'ra' not in d.keys():
		d['ra'],d['dec'] = autil.gal2equ(d['glon'],d['glat'])
	
	
	d['glongc'],d['pzgc'],d['rgc'],d['vlgc'],d['vzgc'],d['vrgc']=autil.helio2gc(d['ra'],d['dec'],d[rkey],d['mura'],d['mudec'],d[rad_vel_key],gal=False,vsun=vsun,usun=usun,xsun=xsun,zsun=zsun,wsun=wsun)
	d['px1'],d['py1'],d['pz1']=autil.lbr2xyz(d['glon'],d['glat'],d[rkey])
	
	l,b,d['mul'],d['mub'] = autil.equ2gal_pm(d['ra'],d['dec'],d['mura'],d['mudec'])
	d['vl']=d['mul']*4.74e3*d[rkey]
	d['vb']=d['mub']*4.74e3*d[rkey]                                               

	d['vx1'],d['vy1'],d['vz1'] = autil.vlbr2xyz(d['glon'],d['glat'],d[rkey],d['vl'],d['vb'],d[rad_vel_key])	

	d['glongc'],d['glatgc'],d['rgcd'],vlgc,vbgc,vrgc=autil.helio2gc(d['ra'],d['dec'],d[rkey],d['mura'],d['mudec'],d[rad_vel_key],gal=False,sph=True,vsun=vsun,usun=usun,xsun=xsun,zsun=zsun,wsun=wsun)                     
	#d['glongc'] = phi_mod(d['glongc'])
	d['pxgc'],d['pygc'],d['pzgc'] = autil.lbr2xyz(d['glongc'],d['glatgc'],d['rgcd'])       
	d['glongc'] = d['glongc']%360.
	
	d['vxgc'],d['vygc'],vzgc = autil.vlbr2xyz(d['glongc'],d['glatgc'],d['rgcd'],d['vlgc'],vbgc,d['vrgc'])	


	
def vgsr(l,b,vhelio,dtype='data'):
	'''
	see Xu 2015, Schonrich 2012
	'''
	b_rad = np.radians(b)
	l_rad = np.radians(l)
	#vgsr = vhelio + (13.84*np.cos(b_rad)*np.cos(l_rad)) + (250.*np.cos(b_rad)*np.sin(l_rad)) + (6.*np.sin(b))	
	vgsr = vhelio + (11.1*np.cos(b_rad)*np.cos(l_rad)) + (8.34*30.24*np.cos(b_rad)*np.sin(l_rad)) + (7.25*np.sin(b))	
	
	if dtype =='galaxia':
		vgsr = vhelio + (11.1*np.cos(b_rad)*np.cos(l_rad)) + (239.08*np.cos(b_rad)*np.sin(l_rad)) + (7.25*np.sin(b))			
	
	return vgsr
def calc_energy(rgc,pzgc,vlgc,vrgc,rscale,vscale,method=''):
	if method == 'galpy':
		print('----------------')
		print('Using galpy.potential')
		print('')

		import galpy		
		from galpy.potential import MWPotential2014				
		energy = galpy.potential.evaluatePotentials(MWPotential2014,rgc/rscale,pzgc/rscale)+0.5*(np.square(vlgc/vscale)+np.square(vrgc/vscale))						
	else:				
		print('----------------')
		print('Using logarithmic approximation')					
		print('')
		
		energy = np.log(rgc/rscale)+0.5*(np.square(vlgc/vscale)+np.square(vrgc/vscale))		
	
	return 	energy	


def n_from_res(tscale,tf,ti,res):
	
	'''
		
    NAME: n_from_res

    PURPOSE: calculate number of orbit integrations to achieve desired resolution
	

    INPUT: 
    
	tf = final time (in Gyr) 
	ti = initial time (in Gyr) 
	res = resolution desired (in Myr)
	
	
    OUTPUT: N 
    
    REMARKS:

       
    HISTORY:   
    
    20 November 2020 (RUG)
    	
	'''


	ti_bv = ti/tscale 
	tf_bv = tf/tscale 			
	
	res_p = res/(tscale*1000.)

	n = (tf_bv-ti_bv)/res_p
	
	
	return int(n)
class orbit_(object):

	def __init__(self,pxgc,pygc,pzgc,vxgc,vygc,vzgc,ti=0.,tf=1,res=20,rscale=8.3,savdir=''):
				
		
		'''
			
	    NAME: orbit_
	
	    PURPOSE: orbit integration
		
	
	    INPUT: 
	    
	    pxgc = Galactocentric X
	    pygc = Galactocentric Y
	    pzgc = Galactocentric Z
	    vxgc = Galactocentric VX
	    vygc = Galactocentric VY
	    vzgc = Galactocentric VZ
	    

		ti = initial time (in Gyr) 
		tf = final time (in Gyr) 		
		res = resolution desired (in Myr)
		
		
	    OUTPUT: N 
	    
	    REMARKS:
	
	       
	    HISTORY:   
	    
	    20 November 2020 (RUG)
	    	
		'''		
		
		print('')
		

		val = locals()		
		self.args={}
		for ky in val.keys():			
			if ky !='self':
				self.args[ky] = val[ky]										



		
		if self.args['savdir'] == '':			
			raise RuntimeError('No savdirectory provided!!')
		print('saving orbits to '+self.args['savdir'])
		
		siminfo={}
		siminfo['rscale']=rscale
		siminfo['vscale']= get_lsr()['omega_sun']*siminfo['rscale']					
		self.tscale = galpy.util.bovy_conversion.time_in_Gyr(siminfo['vscale'],siminfo['rscale'])		

		self.ti_bv = ti/self.tscale 
		self.tf_bv = tf/self.tscale 			
		
		num = n_from_res(self.tscale,tf,ti,res)
		print(num)
		print(tf)
		print(ti)
		siminfo['ts']= np.linspace(self.ti_bv,self.tf_bv,num)
		siminfo['npoints'] = num
		
		self.siminfo = siminfo.copy()
		
		self.prep_()			
		
	def prep_(self):
		'''
		
		
		'''	

		pxgc,pygc,pzgc,vxgc,vygc,vzgc = self.args['pxgc'],self.args['pygc'],self.args['pzgc'],self.args['vxgc'],self.args['vygc'],self.args['vzgc']

		siminfo = self.siminfo.copy()

		
		R = np.sqrt(pxgc**2. + pygc**2.)	
		vlgc,vzgc,vrgc = autil.vxyz2lzr(pxgc,pygc,pzgc,vxgc,vygc,vzgc)
		phi = (np.arctan2(pygc,pxgc))   # in radians
		
		# rescale
		self.R = R/siminfo['rscale']
		self.pzgc = pzgc/siminfo['rscale']
	
		self.vlgc = vlgc/siminfo['vscale']
		self.vrgc = vrgc/siminfo['vscale']
		self.vzgc = vzgc/siminfo['vscale']
		self.phi = phi.copy()
		
		
		# save temporary files
		val_dt = {}
		val_dt['rgc']	= self.R
		val_dt['vrgc']	= self.vrgc
		val_dt['vlgc']	= self.vlgc
		val_dt['vzgc']	= self.vzgc
		val_dt['pzgc']	= self.pzgc
		val_dt['phi']	 = self.phi
		
		ebfwrite(val_dt,'tmp_data_rescaled',self.args['savdir'])		
		ebfwrite(self.siminfo,'tmp_siminfo',self.args['savdir'])		
		
		
		
		
		return 
		
	def intorb_(self):
		
	
		'''
	
	    NAME: orbit_
	
	    PURPOSE: integrate orbits using galpy
		
	
	    INPUT: 
	    
		R = cylindrical Galactocentric radius at which to compute (default=1)
		vrgc = 
		vlgc = 
		pzgc = 
		vzgc 
		phi = 
		
	
	    OUTPUT: density (rgcd)
	    
	    REMARKS:
	    
	    -analytical form taken http://biff.adrian.pw/en/latest/scf/examples.html
	
	    HISTORY:   
	    
	    14 November 2020 (RUG)
	    	    	
		'''
		from galpy.orbit import Orbit
		from galpy.potential import MWPotential2014	

		
		
		ts = self.siminfo['ts']
				
		#i = 0
		#o= Orbit(vxvv=[self.R[i],self.vrgc[i],self.vlgc[i],self.pzgc[i],self.vzgc[i],self.phi[i]])
		#o.integrate(self.siminfo['ts'],MWPotential2014,method='odeint')		
		#o.plot(d1='t',d2='r')
		
		oa=[]
		for i in range(self.R.size):    
			o= Orbit(vxvv=[self.R[i],self.vrgc[i],self.vlgc[i],self.pzgc[i],self.vzgc[i],self.phi[i]])
			oa.append(o)
		
		nsize1=len(oa)
		for i in range(len(oa)):    
			oa[i].integrate(ts,MWPotential2014,method='odeint')
	
		data={}
		data['lgc']=np.array([o.phi(ts) for o in oa])*360.0/(2*np.pi)
		
		# velocities
		data['vlgc']=-np.array([o.vphi(ts)*o.R(ts) for o in oa]) 
		data['vrgc']=np.array([o.vR(ts) for o in oa]) 
		data['vzgc']=np.array([o.vz(ts) for o in oa])	
						
		
		# positions
		data['rgc']=np.array([o.R(ts) for o in oa]) 

		data['pxgc']=np.cos(np.radians(data['lgc']))*data['rgc']
		data['pygc']=np.sin(np.radians(data['lgc']))*data['rgc']	
		data['pzgc']=np.array([o.z(ts) for o in oa]) 

		data['rgcd'] = 	np.sqrt(data['pxgc']**2. + data['pygc']**2. + data['pzgc']**2.)	
		
		
		print('scaling velocity and distance')
		for key in data.keys():
			if key.startswith('v'):
				data[key] = data[key]*self.siminfo['vscale']
			elif key.startswith('r'):
				data[key] = data[key]*self.siminfo['rscale']					
			elif key.startswith('p'):
				data[key] = data[key]*self.siminfo['rscale']					
		

		return data
def intorb_one_(i):
	

	'''

    NAME: orbit_

    PURPOSE: integrate orbits using galpy per star
	

    INPUT: 
    
	R = cylindrical Galactocentric radius at which to compute (default=1)
	vrgc = 
	vlgc = 
	pzgc = 
	vzgc 
	phi = 
	inum = index 
	

    OUTPUT: density (rgcd)
    
    REMARKS:
    

    HISTORY:   
    
    25 November 2020 (RUG)
    	    	
	'''
	from galpy.orbit import Orbit
	from galpy.potential import MWPotential2014	

	hloc = '/net/huygens/data/users/khanna/Documents/pdoc_work/science/halo'

	siminfo = tabpy.read(hloc+'/tmp_orbit/tmp_siminfo.ebf')		
	val_dt = tabpy.read(hloc+'/tmp_orbit/tmp_data_rescaled.ebf')		

	
	
	ts = siminfo['ts']

	
	R = val_dt['rgc']	
	vrgc = val_dt['vrgc']	
	vlgc = val_dt['vlgc']	
	vzgc = val_dt['vzgc']	
	pzgc = val_dt['pzgc']	
	phi = val_dt['phi']	
			
	oa=[]
	o= Orbit(vxvv=[R[i],vrgc[i],vlgc[i],pzgc[i],vzgc[i],phi[i]])
	oa.append(o)
	
	nsize1=len(oa)
	for j in range(len(oa)):    
		oa[j].integrate(ts,MWPotential2014,method='odeint')

	data={}
	data['lgc']=np.array([o.phi(ts) for o in oa])*360.0/(2*np.pi)
	
	# velocities
	data['vlgc']=-np.array([o.vphi(ts)*o.R(ts) for o in oa]) 
	data['vrgc']=np.array([o.vR(ts) for o in oa]) 
	data['vzgc']=np.array([o.vz(ts) for o in oa])	
					
	
	# positions
	data['rgc']=np.array([o.R(ts) for o in oa]) 

	data['pxgc']=np.cos(np.radians(data['lgc']))*data['rgc']
	data['pygc']=np.sin(np.radians(data['lgc']))*data['rgc']	
	data['pzgc']=np.array([o.z(ts) for o in oa]) 

	data['rgcd'] = 	np.sqrt(data['pxgc']**2. + data['pygc']**2. + data['pzgc']**2.)	
	
	
	print('scaling velocity and distance')
	for key in data.keys():
		if key.startswith('v'):
			data[key] = data[key]*siminfo['vscale']
		elif key.startswith('r'):
			data[key] = data[key]*siminfo['rscale']					
		elif key.startswith('p'):
			data[key] = data[key]*siminfo['rscale']					
	
	ebfwrite(data,'orbit_'+str(i),hloc+'/tmp_orbit/orbits')
	return data


class reid_spiral(object):


	def __init__(self,kcor=False):
		print('')
		
		self.kcor = kcor
		self.getarmlist()
	
	def getarmlist(self):
		
		# self.arms = np.array(['3-kpc','Norma','Sct-Cen','Sgt-Car','Local','Perseus','Outer'])
		self.arms = np.array(['3-kpc','Norma','Sct-Cen','Sgr-Car','Local','Perseus','Outer'])
		
	def getparams(self,arm):
		
		if arm == '3-kpc':
			params = {'name':arm,'beta_kink':15,'pitch_low':-4.2,'pitch_high':-4.2,'R_kink':3.52,'beta_min':15,'beta_max':18,'width':0.18}
		if arm == 'Norma':
			params = {'name':arm,'beta_kink':18,'pitch_low':-1.,'pitch_high':19.5,'R_kink':4.46,'beta_min':5,'beta_max':54,'width':0.14}
		if arm == 'Sct-Cen':
			params = {'name':arm,'beta_kink':23,'pitch_low':14.1,'pitch_high':12.1,'R_kink':4.91,'beta_min':0,'beta_max':104,'width':0.23}
		if arm == 'Sgr-Car': #'Sgr-Car'
			params = {'name':arm,'beta_kink':24,'pitch_low':17.1,'pitch_high':1,'R_kink':6.04,'beta_min':2,'beta_max':97,'width':0.27}
		if arm == 'Local':
			params = {'name':arm,'beta_kink':9,'pitch_low':11.4,'pitch_high':11.4,'R_kink':8.26,'beta_min':-8,'beta_max':34,'width':0.31}
		if arm == 'Perseus':
			params = {'name':arm,'beta_kink':40,'pitch_low':10.3,'pitch_high':8.7,'R_kink':8.87,'beta_min':-23,'beta_max':115,'width':0.35}
		if arm == 'Outer':
			params = {'name':arm,'beta_kink':18,'pitch_low':3,'pitch_high':9.4,'R_kink':12.24,'beta_min':-16,'beta_max':71,'width':0.65}
		
		
		if self.kcor:
			Rreid = 8.15
			diffval = params['R_kink'] - Rreid
			xsun = get_lsr()['xsun']
			if diffval < 0:
				 params['R_kink'] = (-xsun) + diffval
			else:
				 params['R_kink'] = (-xsun) + diffval
					
		
		return params


	def model_(self,params):
		'''
		X and Y are flipped in Reid et al. 2019
		I flip it back to sensible orientation here
		
		'''
		
		beta_kink = np.radians(params['beta_kink'])
		pitch_low = np.radians(params['pitch_low'])
		pitch_high = np.radians(params['pitch_high'])
		R_kink = params['R_kink']
		beta_min = params['beta_min']
		beta_max = params['beta_max']
		width = params['width']
		
		
		# beta = np.linspace(beta_min-180,beta_max,100)
		beta = np.linspace(beta_min,beta_max,1000)


		beta_min = np.radians(beta_min)
		beta_max = np.radians(beta_max)
		beta = np.radians(beta)	
		
		
		pitch = np.zeros(beta.size) + np.nan
		indl = np.where(beta<beta_kink)[0]; pitch[indl] = pitch_low
		indr = np.where(beta>beta_kink)[0]; pitch[indr] = pitch_high
		
		tmp1 = (beta - beta_kink)*(np.tan(pitch))
		tmp2 = np.exp(-tmp1)
				
		R = R_kink*tmp2
		x = -R*(np.cos(beta))
		y = R*(np.sin(beta))

		##3 testing 
		R2 = (R_kink+(width*0.5))*tmp2
		x2 = -R2*(np.cos(beta))
		y2 = R2*(np.sin(beta))

		R1 = (R_kink-(width*0.5))*tmp2
		x1 = -R1*(np.cos(beta))
		y1 = R1*(np.sin(beta))
		
		
		
		return x,y, x1,y1,x2,y2
		
	def plot_(self,arm,color='',typ_='HC',xsun_=[],linewidth=0.8,markersize=3,linestyle = '-'):	
		
		if len(xsun_) == 0:
			xsun = get_lsr()['xsun']
		else:
			xsun = xsun_[0]
		
		
		params = self.getparams(arm) ;
		x,y, x1,y1,x2,y2 = self.model_(params);
		if color == '':
			color = 'black'
			
		if typ_ == 'GC':	
			plt.plot(x,y,color,label=params['name'],linestyle='--',linewidth=linewidth)
			plt.plot(x2,y2,color,linestyle='dotted',linewidth=linewidth)
			plt.plot(x1,y1,color,linestyle='dotted',linewidth=linewidth)
			plt.axvline(xsun,linewidth=1,linestyle='--')			
			plt.axhline(0,linewidth=1,linestyle='--')			
			# plt.xlabel('X$_{GC}$')
			# plt.ylabel('Y$_{GC}$')
			plt.plot(0.,0.,marker='+',markersize=10,color='black')
			plt.plot(xsun,0.,marker='o',markersize=10,color='black')
		if typ_ == 'HC':	
			print('..')
			print('using linewidth = '+str(linewidth))
			print('..')
			xhc = x - xsun
			xhc1 = x1 - xsun
			xhc2 = x2 - xsun
			plt.plot(xhc,y,color,label=params['name'],linestyle='-',linewidth=linewidth)
			plt.plot(xhc1,y,color,linestyle='dotted',linewidth=linewidth)
			plt.plot(xhc2,y,color,linestyle='dotted',linewidth=linewidth)
			plt.plot(0.,0.,marker='o',markersize=markersize,color='black')
			plt.plot(-xsun,0.,marker='+',markersize=markersize,color='black')		
			# plt.xlabel('X$_{HC}$')
			# plt.ylabel('Y$_{HC}$')			
			# plt.xlabel('X [kpc]')
			# plt.ylabel('Y [kpc]')			
		
		# plt.legend() 
		
		if typ_ =='polar':
			
			xhc = x - xsun
			xhc1 = x1 - xsun
			xhc2 = x2 - xsun
			
			rgc = sqrtsum(ds=[x,y])
			phi1 = np.arctan2(y,-x)
			phi2 = np.degrees(np.arctan(y/-x))
			phi3 = np.degrees(np.arctan2(y,x))%180.	
			
			# phi3 = 180.-np.degrees(phi1)
			
			# phi1 = (np.arctan2(yhc,xgc))	
			# plt.plot(phi1,rgc,color,linestyle='-',linewidth=linewidth)
			plt.plot(phi1,rgc,'.',color=color,markersize=markersize)
			
			
		if typ_ =='polargrid':
			
			linewidth=2
			
			yhc = y
			xgc = x
			phi4 = np.degrees(np.arctan2(yhc,xgc))%360.	
			rgc = sqrtsum(ds=[x,y])

			plt.plot(np.radians(phi4),rgc,color=color,markersize=markersize,linestyle=linestyle,linewidth=linewidth,label=arm)

	

			
		return 


def spiral_eloisa():

	'''
	plot contours of OB star spirals from Poggio 2021	
	'''	

	pdocdir = getdirec('pdocdir')
	dloc = pdocdir+'/science_verification/DR3/data'
	# #read overdensity contours
	xvalues_overdens=np.load(dloc+'/Eloisa_contours/xvalues_dens.npy')
	yvalues_overdens=np.load(dloc+'/Eloisa_contours/yvalues_dens.npy')
	over_dens_grid=np.load(dloc+'/Eloisa_contours/over_dens_grid_threshold_0_003_dens.npy')
	
	phi1_dens = np.arctan2(yvalues_overdens,-xvalues_overdens)
	Rvalues_dens = sqrtsum(ds=[xvalues_overdens,yvalues_overdens])
	
	# # #------------------ overplot spiral arms in overdens ------------------
	iniz_overdens=0 #.1
	fin_overdens=1.5 #.1
	N_levels_overdens=2
	levels_overdens=np.linspace(iniz_overdens,fin_overdens,N_levels_overdens)
	cset1 = plt.contourf(xvalues_overdens, yvalues_overdens,over_dens_grid.T, levels=levels_overdens,alpha=0.2,cmap='Greys')
	# cset1 = plt.contourf(phi1_dens, Rvalues_dens,over_dens_grid.T, levels=levels_overdens,alpha=0.2,cmap='Greys')
	
	iniz_overdens=0. #.1
	fin_overdens=1.5 #.1
	N_levels_overdens=4#7
	levels_overdens=np.linspace(iniz_overdens,fin_overdens,N_levels_overdens)
	cset1 = plt.contour(xvalues_overdens, yvalues_overdens,over_dens_grid.T, levels=levels_overdens,colors='black',linewidths=0.7)	
	# cset1 = plt.contour(phi1_dens, Rvalues_dens,over_dens_grid.T, levels=levels_overdens,colors='black',linewidths=0.7)	
	
	


class spiral_drimmel(object):
	'''
	
	February 22, 2022
	
	Usage: 
	spi = spiral_drimmel()
	spi.plot_(arm='1',color='b',typ_='HC')
	spi.plot_(arm='2',color='b',typ_='HC')
	spi.plot_(arm='all',color='b',typ_='HC')
	
	'''

	def __init__(self):
		
		
		# self.loc = '/net/huygens/data/users/khanna/Documents/pdoc_work/science_verification/DR3/data/Drimmel_spiral'
		self.loc = getdirec('pdocdir')+'/science_verification/DR3/data/Drimmel_spiral'
		self.fname = 'Drimmel2armspiral.fits'
		self.xsun = get_lsr()['xsun']
		self.getdata(xsun_=[self.xsun])

	def getdata(self,xsun_=[]):
		'''
		
		'''

		dt = tabpy.read(self.loc+'/'+self.fname)
		self.data0 = dt.copy()
		if len(xsun_) == 0:
			xsun = self.xsun
		else:
			xsun = xsun_[0]	
		# rescaling to |xsun|
		qnts = ['rgc1','xhc1','yhc1','rgc2','xhc2','yhc2']
		for qnt in qnts:
			dt[qnt] = dt[qnt]*abs(xsun)		
		
		
		
		#----- add phase-shifted arms as `3` and `4`
		
	
	
		dloc = self.loc+'/phase_shifted'
		for inum in [3,4]:
			dt['xhc'+str(inum)] = np.load(dloc+'/Arm'+str(inum)+'_X_hel.npy')
			dt['yhc'+str(inum)] = np.load(dloc+'/Arm'+str(inum)+'_Y_hel.npy')
			dt['rgc'+str(inum)] = np.sqrt( ((dt['xhc'+str(inum)] + xsun)**2.) + ((dt['yhc'+str(inum)])**2.) )
		
		
		#------------------
		
		
		
		self.data = dt.copy()

		return 
		
	def plot_(self,color='',typ_='HC',xsun_=[],linewidth=0.8,arm='all',markersize=3):	

		if len(xsun_) == 0:
			xsun = get_lsr()['xsun']
		else:
			xsun = xsun_[0]

		self.getdata(xsun_=[xsun])
		dt = self.data.copy()
		# print(list(dt.keys()))
		
		if color == '':
			color = 'black'
					
		numbs = [arm]
		if arm == 'all':
			numbs = ['1','2','4']
			# numbs = ['2','3','4']
		elif arm == 'main':
			numbs = ['1','2']
			# numbs = ['1','4']

		
		
		self.dused = {}
		self.dused['rgc'] = []
		self.dused['xgc'] = []
		self.dused['yhc'] = []
		self.dused['phi1'] = []
		self.dused['phi4'] = []
		
		
		for numb in numbs:
			
			linestyle = '-'
			if float(numb) > 2:
				linestyle = '--'
			
			xhc = dt['xhc'+numb]
			yhc = dt['yhc'+numb]
			rgc = dt['rgc'+numb]
			
			xgc = xhc + xsun
			
			if typ_ == 'HC':	
				
				
				# plt.plot(xhc,yhc,color,label=arm,linestyle=linestyle,linewidth=linewidth,markersize=2)
				plt.plot(xhc,yhc,color,linestyle=linestyle,linewidth=linewidth,markersize=2)
				plt.plot(0.,0.,marker='o',markersize=markersize,color='black')
				plt.plot(-xsun,0.,marker='+',markersize=markersize,color='black')

				# plt.xlabel('X$_{HC}$')
				# plt.ylabel('Y$_{HC}$')			


			
			if typ_ == 'GC':	
				
				# plt.plot(xgc,yhc,color,label=arm,linestyle=linestyle,linewidth=linewidth)
				plt.plot(xgc,yhc,color,linestyle=linestyle,linewidth=linewidth)
				plt.axvline(xsun,linewidth=1,linestyle='--')			
				plt.axhline(0,linewidth=1,linestyle='--')			
				# plt.xlabel('X$_{GC}$')
				# plt.ylabel('Y$_{GC}$')
				plt.plot(0.,0.,marker='+',markersize=10,color='black')
				plt.plot(xsun,0.,marker='o',markersize=10,color='black')
				self.dused['xgc'].append(xgc)
				self.dused['yhc'].append(yhc)
		

			if typ_ =='polar':
				

				phi1 = np.arctan2(yhc,-xgc)
				
				phi2 = np.degrees(np.arctan(yhc/-xgc))
				phi3 = np.degrees(np.arctan2(yhc,xgc))%180.	
				phi4 = np.degrees(np.arctan2(yhc,xgc))%360.	
				
				# phi3 = 180.-np.degrees(phi1)
				
				# phi1 = (np.arctan2(yhc,xgc))	
				# plt.plot(np.degrees(phi1),rgc,color,linestyle='--',linewidth=linewidth)
				# plt.plot(np.degrees(phi1),rgc,'.',color='blue')
				plt.plot(phi1,rgc,color=color,markersize=markersize,linestyle=linestyle,linewidth=linewidth)

				self.dused['rgc'].append(rgc)
				self.dused['xgc'].append(xgc)
				self.dused['yhc'].append(yhc)
				self.dused['phi1'].append(phi1)
				self.dused['phi4'].append(phi4)
				
				
			if typ_ =='polargrid':
				

				phi1 = np.arctan2(yhc,-xgc)
				
				phi2 = np.degrees(np.arctan(yhc/-xgc))
				phi3 = np.degrees(np.arctan2(yhc,xgc))%180.	
				phi4 = np.degrees(np.arctan2(yhc,xgc))%360.	
				
				# phi3 = 180.-np.degrees(phi1)


				if numb == numbs[0]:
					plt.plot(np.radians(phi4),rgc,color=color,markersize=markersize,linestyle=linestyle,linewidth=linewidth,label='NIR')

				else:
					plt.plot(np.radians(phi4),rgc,color=color,markersize=markersize,linestyle=linestyle,linewidth=linewidth)
					



				self.dused['rgc'].append(rgc)
				self.dused['xgc'].append(xgc)
				self.dused['yhc'].append(yhc)
				self.dused['phi1'].append(phi1)
				self.dused['phi4'].append(phi4)
				
				
				# plt.plot(xgc,yhc,'.',color=color)
		
		# plt.legend() 
		return 

	

class spiral_cepheids(object):
	'''
	
	June 28, 2024
	
	Usage: 
	spi = spiral_drimmel()
	spi.plot_(arm='1',color='b',typ_='HC')
	spi.plot_(arm='2',color='b',typ_='HC')
	spi.plot_(arm='all',color='b',typ_='HC')
	
	'''

	def __init__(self):
	
		
		# where the data is
		
		self.pdocdir = getdirec('pdocdir')		
		self.cephloc = self.pdocdir+'/science/dr3/data/cepheids'	
		self.spiral_loc = self.cephloc+'/spiral_model'
			
		
		self.fname = 'ArmAttributes_dyoungW1_bw025.pkl'

		self.spirals = pickleread(self.spiral_loc+'/'+self.fname)
	
		self.armlist = list(self.spirals['0']['arm_attributes'].keys())
		
		self.xsun = get_lsr()['xsun'] 
		self.rsun = get_lsr()['Rsun'] 

		
	def plotit_(self,armplt='',typ_='GC',markersize=4,linewidth=2,linestyle2='--'):
		
		from time import sleep
		from scipy.signal import find_peaks
		import numpy as np
		import astropy
		import astropy.table as tb
		import pandas as pd
		import os
		import matplotlib as mpl
		import matplotlib.pyplot as plt
		import imp 
		import dtools
		import autil 
		import tabpy
		params = {'font.size':12,
		      'text.usetex':False,
		      'ytick.labelsize': 'medium',
		      'legend.fontsize': 'large',
		      'axes.linewidth': 1.0,
		      'figure.dpi': 150.0,
		      'xtick.labelsize': 'medium',
		      'font.family': 'sans-serif',
		      'axes.labelsize': 'medium'}
		mpl.rcParams.update(params)
		
		imp.reload(dtools)
		

		
		spirals = self.spirals
		
		# arms and plotting colors for the arms
		colors = ['C3','C0','C1','C2']
		arms = np.array(['Scutum','Sag-Car','Orion','Perseus'])
		
		figtyp = 'png'
		
		# XY positions
		rsun = self.rsun  # Might want to put these in a configuration file
		xsun = self.xsun  # Might want to put these in a configuration file
		lnrsun = np.log(rsun) 
		
		
		# best phi range:
		phi_range = np.deg2rad(np.sort(spirals['1']['phi_range'].copy()))
		maxphi_range = np.deg2rad([60,-120]) 
		
		
		# arms and plotting colors for the arms
		colors = ['C3','C0','C1','C2']
		arms = np.array(self.armlist)
		
		arm_clr = {'Scutum':'C3','Sag-Car':'C0','Orion':'C1','Perseus':'C2'}
		self.arm_clr = arm_clr
		
		dt = {}
		
		
		for armi in np.arange(arms.size):
			
			arm = arms[armi]
			pang = (spirals['1']['arm_attributes'][arm]['arm_pang_strength']+spirals['1']['arm_attributes'][arm]['arm_pang_prom'])/2.
			lnr0 = (spirals['1']['arm_attributes'][arm]['arm_lgr0_strength']+spirals['1']['arm_attributes'][arm]['arm_lgr0_prom'])/2.
			
						
			
			# plot the arms
			
			
			phi=(np.arange(51)/50.)*np.diff(phi_range)[0] + phi_range[0]  
			lgrarm = lnr0 - np.tan(np.deg2rad(pang))*phi 
			
			
			xgc = -np.exp(lgrarm)*np.cos(phi); xhc = xgc - xsun
			ygc = np.exp(lgrarm)*np.sin(phi) ;  yhc = ygc
			
							
			# extrapolate the arms
			phi=(np.arange(101)/100.)*np.diff(maxphi_range)[0] + maxphi_range[0]  
			lgrarm = lnr0 - np.tan(np.deg2rad(pang))*phi 
			
			xgc_ex = -np.exp(lgrarm)*np.cos(phi);  xhc_ex = xgc_ex - xsun
			ygc_ex = np.exp(lgrarm)*np.sin(phi); yhc_ex = ygc_ex
			lonarm = np.arctan((np.exp(lgrarm)*np.sin(phi))/(rsun - np.exp(lgrarm)*np.cos(phi))) 
			
			dt[arm] = {}
			dt[arm]['xgc'] = xgc
			dt[arm]['xhc'] = xhc
			dt[arm]['ygc'] = ygc
			dt[arm]['yhc'] = yhc
						
			dt[arm]['xgc_ex'] = xgc_ex
			dt[arm]['xhc_ex'] = xhc_ex
			dt[arm]['ygc_ex'] = ygc_ex
			dt[arm]['yhc_ex'] = yhc_ex
						


		self.dused = {}
		self.dused['rgc'] = []
		self.dused['xgc'] = []
		self.dused['yhc'] = []
		self.dused['phi1'] = []
		self.dused['phi4'] = []
		


		# for armi in dt.keys():
		for armi in armplt:
			# print(armplt)
						
			xgc = dt[armi]['xgc']
			ygc = dt[armi]['ygc']
			xhc = dt[armi]['xhc']
			yhc = dt[armi]['yhc']
			
			xgc_ex = dt[armi]['xgc_ex']
			ygc_ex = dt[armi]['ygc_ex']
			xhc_ex = dt[armi]['xhc_ex']
			yhc_ex = dt[armi]['yhc_ex']			


			rgc = np.sqrt(xgc**2. + ygc**2.)
			rgc_ex = np.sqrt(xgc_ex**2. + ygc_ex**2.)



			if typ_ == 'GC':
				

				
				plt.plot(xgc,ygc,'-',color=arm_clr[armi])				
				plt.plot(xgc_ex,ygc_ex,linestyle=linestyle2,color=arm_clr[armi])
				
				plt.axvline(xsun,linewidth=1,linestyle='--')			
				plt.axhline(0,linewidth=1,linestyle='--')			
				
	
				plt.plot(0.,0.,marker='+',markersize=10,color='black')
				plt.plot(xsun,0.,marker='o',markersize=10,color='black')				
	
				plt.xlabel('X$_{GC}$')
				plt.ylabel('Y$_{GC}$')
				    
				
				
				
				# # labels
				# plt.text(-2,-5,'Scutum',fontsize=14,fontweight='bold',color=colors[0])
				# plt.text(-2,-10,'Sgr-Car',fontsize=14,fontweight='bold',color=colors[1])
				# plt.text(-3,-13,'Local',fontsize=14,fontweight='bold',color=colors[2])
				# plt.text(-9,-14,'Perseus',fontsize=14,fontweight='bold',color=colors[3])		
	    
	    
			if typ_ == 'HC':	
				
				
		
			
				
				# plt.plot(xhc,yhc,color,label=arm,linestyle=linestyle,linewidth=linewidth,markersize=2)
				# plt.plot(xhc,yhc,color,linestyle=linestyle,linewidth=linewidth,markersize=2)
				
				
				plt.plot(xhc,yhc,'-',color=arm_clr[armi])				
				plt.plot(xhc_ex,yhc_ex,linestyle=linestyle2,color=arm_clr[armi])
	
				plt.plot(0.,0.,marker='o',markersize=markersize,color='black')
				plt.plot(-xsun,0.,marker='+',markersize=markersize,color='black')						
								
	
				
				plt.xlabel('X$_{HC}$')
				plt.ylabel('Y$_{HC}$')			
	
	
	
	
			if typ_ =='polar':
							
				
				phi1 = np.arctan2(yhc,-xgc)
				phi1_ex = np.arctan2(ygc_ex,-xgc_ex)
				
				phi2 = np.degrees(np.arctan(yhc/-xgc))
				phi3 = np.degrees(np.arctan2(yhc,xgc))%180.	
				phi4 = np.degrees(np.arctan2(yhc,xgc))%360.	
				
				plt.plot(phi1,rgc,'-',color=arm_clr[armi],markersize=markersize)
				plt.plot(phi1_ex,rgc_ex,linestyle=linestyle2,color=arm_clr[armi],markersize=markersize)
			
						    
				self.dused['rgc'].append(rgc)
				self.dused['xgc'].append(xgc)
				self.dused['yhc'].append(yhc)
				self.dused['phi1'].append(phi1)
				self.dused['phi4'].append(phi4)


			if typ_ =='polargrid':
				
				linewidth=2
				linestyle = '-'
				phi4 = np.degrees(np.arctan2(yhc,xgc))%360.	
				phi4_ex = np.degrees(np.arctan2(ygc_ex,xgc_ex))%360.	

				phi1 = np.arctan2(yhc,-xgc)
				phi1_ex = np.arctan2(ygc_ex,-xgc_ex)
	

				# plt.plot(np.radians(phi4),rgc,color=arm_clr[armi],markersize=markersize,linestyle=linestyle,linewidth=linewidth)

				# plt.plot(phi1,rgc,color=arm_clr[armi],markersize=markersize,linestyle=linestyle,linewidth=linewidth)
				# plt.plot(phi1_ex,rgc_ex,color=arm_clr[armi],markersize=markersize,linestyle='--',linewidth=linewidth)
				
				plt.plot(np.radians(phi4),rgc,color=arm_clr[armi],markersize=markersize,linestyle=linestyle,linewidth=linewidth)
				
				plt.plot(np.radians(phi4_ex),rgc_ex,color=arm_clr[armi],markersize=markersize,linestyle=linestyle2,linewidth=linewidth)
				# plt.plot(np.radians(phi4_ex),rgc_ex,color=arm_clr[armi],markersize=markersize,linestyle='--',linewidth=linewidth)

	
				self.dused['rgc'].append(rgc)
				self.dused['xgc'].append(xgc)
				self.dused['yhc'].append(yhc)
				self.dused['phi4'].append(phi4)
				
					


	
def ovplot_mw():

	'''
	under development....
	plot the Milky Way background using Leung's package
	
	
	History : created on July 24, 2023 [INAF, Torino]
	
	'''

	import pylab as plt
	
	from mw_plot import MWPlot, MWSkyMap
	
	from astropy import units as  u
	
	
	# setup a mw-plot instance of bird's eyes view of the disc
	mw2 = MWPlot(radius=20 * u.kpc, center=(0, 0)*u.kpc, unit=u.kpc, coord='galactocentric', rot90=2, grayscale=True, annotation=False)
	mw3 = MWSkyMap()
	
	# setup subplots with matplotlib
	
	fig = plt.figure(figsize=(5,5))
	
	ax1 = fig.add_subplot(221)
	
	ax2 = fig.add_subplot(222)
	
	# transform the subplots with different style
	
	mw2.transform(ax1)
	
	mw3.transform(ax2)
	
	
	fig.tight_layout()


	return 




class sampfromcov_gaia(object):

	'''
    NAME:

    PURPOSE:
	 - Constructs covariance matrix for Gaia data. 
	 - Used to generate samples from multivariate Gaussian


    INPUT: data as dictionary

    OUTPUT:
       
    HISTORY:        
      
    
    October 07, 2021: 
    October 30, 2024: INAF
		
	'''
	
	def __init__(self,data1):		
				
		
		self.dtrim, self.drest = self.prep_from_samp(data1)		
		data = self.dtrim.copy()
		
				
		# print('777')
				
		data['pmra'] = data['mura']*1e+03
		data['pmdec'] = data['mudec']*1e+03

		self.data = Table(data)		
		self.dsize = data['pmra'].size

		# Create a copy of the default unit map		
		self.get_unit_map()

		# Store the source table
		data = self.data 	

		self.allnames = ['ra', 'dec', 'parallax', 'pmra', 'pmdec','radial_velocity']		
		
		# Update the unit map with the table units
		self._invalid_units = dict()
		for c in data.colnames:
			if data[c].unit is not None:
				try:
					self.units[c] = u.Unit(str(data[c].unit))
				except ValueError:
					self._invalid_units[c] = data[c].unit		
		
				
		
		self.has_rv = 'radial_velocity' in self.data.colnames
		self._cache = dict()



	def prep_from_samp(self,data1):
		
		'''
		input: the entire dataset
		
		dtrim = dictionary containing only relevant keys
		drest = dictionary containing remaining keys to be joined later on
		
		'''
		
			
	
		restkys = []			
		self.limnames = ['ra', 'dec', 'parallax', 'pmra', 'pmdec','radial_velocity','mura','mudec']		
		for ky in data1.keys():				
			if 'pseudocolour' not in ky and 'phot' not in ky and 'feh' not in ky:
				if 'corr' in ky or 'error' in ky:
				
					self.limnames.append(ky)
		
		for ky in data1.keys():				
			if ky not in self.limnames:
				restkys.append(ky)
	
		restkys = np.array(restkys)
				
		dtrim = {}
		for ky in self.limnames:
			dtrim[ky] = data1[ky].copy()	
	
		drest = {}
		for ky in restkys:
			drest[ky] = data1[ky].copy()	
			
		
		
		dtrim['source_id'] = drest['source_id'].copy()
		
		

		
		self.dtrim = dtrim.copy()	
		self.drest = drest.copy()	
		
		
			
		return self.dtrim, self.drest




	def get_unit_map(self):
		import astropy.units as u
		gaia_unit_map = {
		    'ra': u.degree,
		    'dec': u.degree,
		    'parallax': u.milliarcsecond,
		    'pmra': u.milliarcsecond / u.year,
		    'pmdec': u.milliarcsecond / u.year,
		    'radial_velocity': u.km / u.s,
		    'ra_error': u.milliarcsecond,
		    'dec_error': u.milliarcsecond,
		    'parallax_error': u.milliarcsecond,
		    'pmra_error': u.milliarcsecond / u.year,
		    'pmdec_error': u.milliarcsecond / u.year,
		    'radial_velocity_error': u.km / u.s,
		    'astrometric_excess_noise': u.mas,
		    'astrometric_weight_al': 1/u.mas**2,
		    'astrometric_pseudo_colour': 1/u.micrometer,
		    'astrometric_pseudo_colour_error': 1/u.micrometer,
		    'astrometric_sigma5d_max': u.mas,
		    'phot_g_mean_flux': u.photon/u.s,
		    'phot_g_mean_flux_error': u.photon/u.s,
		    'phot_g_mean_mag': u.mag,
		    'phot_bp_mean_flux': u.photon/u.s,
		    'phot_bp_mean_flux_error': u.photon/u.s,
		    'phot_bp_mean_mag': u.mag,
		    'phot_rp_mean_flux': u.photon/u.s,
		    'phot_rp_mean_flux_error': u.photon/u.s,
		    'phot_rp_mean_mag': u.mag,
		    'bp_rp': u.mag,
		    'bp_g': u.mag,
		    'g_rp': u.mag,
		    'rv_template_teff': u.K,
		    'l': u.degree,
		    'b': u.degree,
		    'ecl_lon': u.degree,
		    'ecl_lat': u.degree,
		    'teff_val': u.K,
		    'teff_percentile_lower': u.K,
		    'teff_percentile_upper': u.K,
		    'a_g_val': u.mag,
		    'a_g_percentile_lower': u.mag,
		    'a_g_percentile_upper': u.mag,
		    'e_bp_min_rp_val': u.mag,
		    'e_bp_min_rp_percentile_lower': u.mag,
		    'e_bp_min_rp_percentile_upper': u.mag,
		    'radius_val': u.Rsun,
		    'radius_percentile_lower': u.Rsun,
		    'radius_percentile_upper': u.Rsun,
		    'lum_val': u.Lsun,
		    'lum_percentile_lower': u.Lsun,
		    'lum_percentile_upper': u.Lsun,
		    'ref_epoch': u.year
		}
			
		self.units = gaia_unit_map.copy()
		self.gaia_unit_map = gaia_unit_map.copy()
		
		
	def get_cov(self,units=None):
		"""
		The Gaia data tables contain correlation coefficients and standard
		deviations for (ra, dec, parallax, pm_ra, pm_dec), but for most analyses
		we need covariance matrices. This converts the data provided by Gaia
		into covariance matrices.
	
		If a radial velocity exists, this also contains the radial velocity
		variance. If radial velocity doesn't exist, that diagonal element is set
		to inf.
	
		The default units of the covariance matrix are [degree, degree, mas,
		mas/yr, mas/yr, km/s], but this can be modified by passing in a
		dictionary with new units. For example, to change just the default ra,
		dec units for the covariance matrix, you can pass in::
	
			units=dict(ra=u.radian, dec=u.radian)
	
		Parameters
		----------
		RAM_threshold : `astropy.units.Quantity`
			Raise an error if the expected covariance array is larger than the
			specified threshold. Set to ``None`` to disable this checking.
		"""
	
		# The full returned matrix
		C = np.zeros((len(self.data), 6, 6))
	
		# We handle radial_velocity separately below - doesn't have correlation
		# coefficients with the astrometric parameters
		names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
		

		# pre-load the diagonal
		for i, name in enumerate(names):
			if name + "_error" in self.data.colnames:
				err = (self.data[name + "_error"])*self.gaia_unit_map[name + "_error"]
				C[:, i, i] = err.to(self.gaia_unit_map[name]).value ** 2
			else:
				C[:, i, i] = np.nan
	
		if self.has_rv:
			name = 'radial_velocity'
			err = self.data[name + "_error"]			
			C[:, 5, 5] = err ** 2
		else:
			C[:, 5, 5] = np.inf
	
		C[:, 5, 5][np.isnan(C[:, 5, 5])] = np.inf # missing values
	
		for i, name1 in enumerate(names):
			for j, name2 in enumerate(names):
				if j <= i:
					continue

				if "{0}_{1}_corr".format(name1, name2) in self.data.colnames:
					corr = self.data["{0}_{1}_corr".format(name1, name2)]
				else:
					corr = np.nan

				# We don't need to worry about units here because the diagonal
				# values have already been converted
				C[:, i, j] = corr * np.sqrt(C[:, i, i] * C[:, j, j])
				C[:, j, i] = C[:, i, j]
	
		self._cache['cov'] = C
		self._cache['cov_units'] = units
	
		return self._cache['cov']

	def get_error_samples(self, size=1, rnd=None):		
		"""Generate a sampling from the Gaia error distribution for each source.
	
		This function constructs the astrometric covariance matrix for each
		source and generates a specified number of random samples from the error
		distribution for each source. This does not handle spatially-dependent
		correlations. Samplings generated with this method can be used to, e.g.,
		propagate the Gaia errors through coordinate transformations or
		analyses.
	
		Parameters
		----------
		size : int
			The number of random samples per soure to generate.
		rnd : ``numpy.random.RandomState``, optional
			The random state.
	
		Returns
		-------
		g_samples : `pyia.GaiaData`
			The same data table, but now each Gaia coordinate entry contains
			samples from the error distribution.
	
		"""

		if rnd is None:
			rnd = np.random.RandomState()

		tn = self.get_cov()		
		rv_mask = ~np.isfinite(tn[:, 5, 5])
		tn[rv_mask, 5, 5] = 0.		
		
		arrs = []
		for nm in self.allnames:
			arrs.append(self.data[nm])
		y = np.stack(arrs).T
		rnd = np.random.RandomState()		
		
		samples = np.array([rnd.multivariate_normal(y[i], tn[i], size=size) for i in range(len(y)) ])
	
		
		
		dtemp = {}
		for ky in self.allnames:
			dtemp[ky] = np.zeros(self.dsize) + np.nan
		
		for jnum in range(self.dsize):
			for inum,ky in enumerate(self.allnames):					
				dtemp[ky][jnum] = samples[jnum][0][inum]
			
		return dtemp






# Gaia validation

def get_gaiaxpy():
	'''
	to import bmcmc2 from external directory
	01/Nov/2018
	
	'''
	import sys	
	#loc = '/home/shourya/Documents/phd_work'
	loc = getdirec('pdocdir')+'/validation/gaia_validation/GaiaXPy'
	
	sys.path.insert(0,loc)
	import gaiaxpy as gxpy
	
	return gxpy

def get_gunlim():
	'''
	to import gaiaunlimited from external directory
	10/Mar/2023
	
	'''
	import sys	
	#loc = '/home/shourya/Documents/phd_work'
	loc = getdirec('pdocdir')+'/gaiaunlimited/src'
	
	sys.path.insert(0,loc)
	import gaiaunlimited as gunlim
	
	return gunlim


def make_kld_readme():

	'''	
		
    NAME: make_kld_readme

    PURPOSE: makes readme files for kld runs


    INPUT: - keyboard input

    OUTPUT: saves file in valdir
       
    HISTORY: March 07, 2021 (Groningen)       
		
	
	'''

	pdocdir = getdirec('pdocdir')
	valdir = pdocdir+'/validation/gaia_validation'



	dt = {}

	
	date = input('month_date_year = ')
	data = input('data ex: sources_03_03.... = ')
	subset = input('subset = ')
	keys_run = input('keys_run = ')
	comments = input('comments = ')
	
	dt['date']  = [date]	
	dt['data'] = [data]
	dt['subset'] = [subset]
	dt['keys_run'] = [keys_run]
	dt['comments'] = [comments]
	
	
	
	tab = Table(dt)
	tab.write(valdir+'/readme.csv',format='csv') 
	
	print('readme.csv file written in .... '+valdir)
	
	return 



# Galaxia
def add_kinematics(d,thick=True,modify=True):
	
	if modify:
		usun=11.1
		vsun=239.08
		wsun=7.25
		xsun=-8.0
		zsun=0.0
		tmax=10.0; tmin=0.1
		Rd=2.5
		Rsig0=1/0.073
		Rsig1=1/0.132
		sigma_phi0=25.34
		sigma_z0=25.92
		sigma_R0=36.69
		betaR=0.164
		betaphi=0.164
		betaz=0.37
		
		sigma_phi1=38.37
		sigma_z1=39.41
		sigma_R1=57.87
		
		home = getdirec('home')	
		#vc_loc = '/work1/sharma/GsynthData'
		vc_loc = getdirec('galaxia')+'/Model' 
		vcirc1=tab.read(vc_loc+'/vcirc_potential.csv')

	
		glongc,pzgc,rgc=autil.xyz2lzr(d['px']+xsun,d['py'],d['pz']+zsun)
		age=np.power(10.0,d['age']-9.0)
		
		Rsig=d['px']-d['px']+Rsig0
		fac=np.exp(-(rgc+xsun)/Rsig)
		sigma_phi=sigma_phi0*np.power((age+tmin)/(tmax+tmin),betaphi)*fac
		sigma_z=sigma_z0*np.power((age+tmin)/(tmax+tmin),betaz)*fac
		sigma_R=sigma_R0*np.power((age+tmin)/(tmax+tmin),betaR)*fac
		
		if thick:
			ind=np.where(d['popid']>6)[0]
			Rsig[ind]=Rsig1
			fac[ind]=np.exp(-(rgc[ind]+xsun)/Rsig[ind])
			sigma_phi[ind]=sigma_phi1*fac[ind]
			sigma_z[ind]==sigma_z1*fac[ind]
			sigma_R[ind]==sigma_R1*fac[ind]
			
		ind_bulge_etc = np.where(d['popid'] > 7)[0] 	
		vx_bluge_etc,vy_bluge_etc,vz_bluge_etc =  d['vx'][ind_bulge_etc],d['vy'][ind_bulge_etc],d['vz'][ind_bulge_etc]				
		
		#vcirc=239.08-12.2 
		vcirc= np.interp(rgc,vcirc1['r'],vcirc1['vcirc'])
		temp=vcirc*vcirc+sigma_R*sigma_R*(-rgc/Rd-2*rgc/Rsig+1-np.power(sigma_phi/sigma_R,2.0)+(1-np.power(sigma_z/sigma_R,2.0))*0.5)
		print (np.mean(sigma_R),np.mean(-np.sqrt(temp)),np.mean(vcirc),np.mean(age))
		vphi=-np.sqrt(temp)+np.random.normal(size=d['px'].size)*sigma_phi
		vz=np.random.normal(size=d['px'].size)*sigma_z
		vR=np.random.normal(size=d['px'].size)*sigma_R
		l,b,r,vl,vb,vr=autil.gc2helio(glongc,pzgc,rgc,vphi,vz,vR,gal=True,vsun=vsun,usun=usun,xsun=xsun)
		d['vx'],d['vy'],d['vz']=autil.vlbr2xyz(d['glon'],d['glat'],d['rad'],vl,vb,vr)
		
		d['vx'][ind_bulge_etc],d['vy'][ind_bulge_etc],d['vz'][ind_bulge_etc] = vx_bluge_etc,vy_bluge_etc,vz_bluge_etc
		
	gutil.append_muvr(d)
def getmtip(age,feh):

	'''
    NAME: make_kld_readme

    PURPOSE: makes readme files for kld runs


    INPUT: age [Gyr], [Fe/H]

    OUTPUT: mtip
       
    HISTORY: July 20, 2021 (Groningen)       
	
	'''

	loc = getdirec('galaxia')
	d = ebf.read(loc+'/feh_age_mtip.ebf','/')
	sgrid=sutil.InterpGrid(tabpy.npstruct(d),['feh','log_age'])
	log_age = np.log10((age)*1e+09)
	
	mtip = sgrid.get_values('mtip',[feh,log_age],fill_value='nearest')

	return mtip


class mkglx(object):

	'''
	to make magnitude limited surveys from a galaxia file etc
	
	february 17, 2023 [INAF-TORINO]

	'''	
	def __init__(self,data=None,writehere=''):
		
		
		self.writehere = writehere
		self.glxdir = getdirec('galaxia')
	
	def generate_(self,runfname='galaxy1'):
		
		
		
		
		os.system('galaxia -r '+runfname+'.ebf')
		os.system('galaxia -a --psys=GAIA '+runfname+'.ebf')
		
		
		
		return
		
	 
	def mini_file_old(self,survey='galaxia',add_errors=False,fac=1.0,iso='tmasswise'):	
		
		# mydata =  tabpy.read(desktop+'/galaxia_nw_allsky_rc_f0.01.ebf')
		
		# downloads = dtools.getdirec('downloads')
		# fname = 'hermes4_nw_tmasswise_fehm0.3_jv15_f1.0_cut.ebf'
		
		# mydata = tabpy.read(downloads+'/'+fname)
		
		mydata = tabpy.read(self.glxdir+'/runs_/'+self.fname_+'.ebf')
		
		if 'log' in mydata.keys():
			del mydata['center']
			del mydata['log']
	
		print(mydata['pz'].size,survey+' stars in all')		
	
		# high [Fe/H]	
		ind = np.where(mydata['feh'] > -2.0)[0]	
		print(str(len(ind))+' high [Fe/H] stars')
		high_data = dtools.filter_dict(mydata,ind)
		crnt = (high_data['px'].size)*fac; crnt = int(crnt)	
		tabpy.shuffle(high_data,crnt)
		print(high_data['pz'].size,survey+' [Fe/H] > -2.0 stars shuffled')		
		
		# low [Fe/H]
		ind = np.where(mydata['feh'] < -2.0)[0]		
		mydata = dtools.filter_dict(mydata,ind)
		print(str(mydata['px'].size)+' low [Fe/H] stars')
		
		tabpy.union(mydata,high_data,pkey=None)
		
		print(mydata['pz'].size,survey+' stars selected')
		if add_errors:
			print('adding photometric & spectroscopic errors')
			
			gutil.add_tmass_ph_error(mydata)
			add_spec_error(mydata)	
		
	
		mydata['gaia_g_1'],mydata['gaia_gbp_1'],mydata['gaia_grp_1'] = mydata['gaia_g'],mydata['gaia_gbp'],mydata['gaia_grp']	
		mydata[iso+'_h_1'],mydata[iso+'_j_1'],mydata[iso+'_ks_1'] = mydata[iso+'_h'],mydata[iso+'_j'],mydata[iso+'_ks']	
		mydata[iso+'_w1_1'],mydata[iso+'_w2_1'],mydata[iso+'_w3_1'],mydata[iso+'_w4_1'] = mydata[iso+'_w1'],mydata[iso+'_w2'],mydata[iso+'_w3'],mydata[iso+'_w4']
		gutil.abs2app(mydata,isoch_loc=dtools.getdirec('galaxia'),noext=False,corr=False)
		#gutil.append_muvr(mydata)
		
		mydata['teff'] = 10**mydata['teff']
		mydata['logg'] = mydata['grav'].copy()
		
		if add_errors:
			ebf.write(root_+'/glxminifile_errors.ebf','/',mydata,'w')			
		else:
			ebf.write(root_+'/glxminifile.ebf','/',mydata,'w')	
		
		return mydata
		
	def reducecols(self,useloc='',runfname='galaxy1',svfname='galaxy1_red'):
		
		
		keepkys = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'feh', 'smass', 'age', 'rad',
		       'popid', 'lum', 'teff', 'grav', 'mact', 'mtip', 'tmasswise_j',
		       'tmasswise_h', 'tmasswise_ks', 'tmasswise_w1', 'tmasswise_w2',
		       'tmasswise_w3', 'tmasswise_w4', 'exbv_schlegel', 'exbv_solar',
		       'exbv_schlegel_inf', 'glon', 'glat', 'gaia_g', 'gaia_gbp',
		       'gaia_grp']		
		
		if useloc =='':
			
			mydata = dtools.ebfread_keys(self.glxdir+'/runs_/'+runfname+'.ebf',keys=keepkys)
		else:
			mydata = dtools.ebfread_keys(useloc,keys=keepkys)			
		
		dtools.ebfwrite(mydata,svfname,self.glxdir)
		
		return
	
	def mini_file(self,runfname='galaxy1_red',survey='galaxia',add_errors=False,fac=1.0,iso='tmasswise',noext=False,maglimband='',maglim=10000):	
	
		suff_ = ''
		if noext:
			suff_ = '_noext'
		print('noext is...'+str(noext))
	

		mydata = tabpy.read(self.glxdir+'/'+runfname+'.ebf')
	
		
		
		if 'log' in mydata.keys():
			del mydata['center']
			del mydata['log']
	
		print(mydata['pz'].size,survey+' stars in all')		
		
		if fac < 1.:
			mk_mini(mydata,fac=fac,key='pz')
			
	

		if add_errors:
			print('adding photometric & spectroscopic errors')
			
			gutil.add_tmass_ph_error(mydata)
			add_spec_error(mydata)	
	
		mydata['gaia_g_1'],mydata['gaia_gbp_1'],mydata['gaia_grp_1'] = mydata['gaia_g'],mydata['gaia_gbp'],mydata['gaia_grp']	
		mydata[iso+'_h_1'],mydata[iso+'_j_1'],mydata[iso+'_ks_1'] = mydata[iso+'_h'],mydata[iso+'_j'],mydata[iso+'_ks']	
		mydata[iso+'_w1_1'],mydata[iso+'_w2_1'],mydata[iso+'_w3_1'],mydata[iso+'_w4_1'] = mydata[iso+'_w1'],mydata[iso+'_w2'],mydata[iso+'_w3'],mydata[iso+'_w4']
		gutil.abs2app(mydata,isoch_loc=dtools.getdirec('galaxia'),noext=noext,corr=False)
		#gutil.append_muvr(mydata)
		
		mydata['teff'] = 10**mydata['teff']
		mydata['logg'] = mydata['grav'].copy()
		del mydata['grav']
		
		
		if maglimband != '':
			
			tabpy.where(mydata,condition=(mydata[maglimband]<maglim))
		
		if add_errors:
			ebf.write(self.writehere+'/glxminifile_errors.ebf','/',mydata,'w')			
		else:
			dtools.ebfwrite(mydata,'glxminifile'+suff_,self.writehere)
		
		return mydata
		



#--- Data prep
def loadgaia(data,quasar_offset=True,p_errorcut=0.2):
	
	print('Loading GAIA data.....')
	
	
	loc = getdirec('phddir')+'/gaia/data'
	#d = tabpy.read(loc+'/gaiadata_use_allkeys.ebf')
	#d = tabpy.read(loc+'/gaiadata_use.ebf')
	#d = tabpy.read(loc+'/2massgaia.fits')
	
	d = data.copy()
	
	d['mura'] = d['pmra']*1e-03
	d['mura_er'] = d['pmra_error']*1e-03	
	d['mudec'] = d['pmdec']*1e-03
	d['mudec_er'] = d['pmdec_error']*1e-03	
	
	d['vr'] = d['radial_velocity']
	d['vr_er'] = d['radial_velocity_error']	

	keylist = ['l','b','ra','dec','mura','mura_er','mudec','mudec_er','vr','vr_er','parallax','parallax_error','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','a_g_val',
	            'bp_g','bp_rp','g_rp','e_bp_min_rp_val','source_id']

	for key in d.keys():				
		if key not in keylist:						
			del d[key]
						
	# derived keys
	
	print('Total stars: '+str(d['ra'].size))	
	tabpy.where(d,condition=(d['parallax']>0.)		
							 &(np.isfinite(d['parallax']))
	                         &((d['parallax_error']/d['parallax'])<p_errorcut))

	if quasar_offset:		
		print('Applying quasar offset of +0.029 mas')
		d['parallax'] = d['parallax'] + 0.029		
	else:
		print('No quasar offset of +0.029 mas')	
	d['dist_gaia'] = 1.0/d['parallax']
	
	d['glon'],d['glat']=autil.equ2gal(d['ra'],d['dec'])
	
	print('Total selected stars: '+str(d['ra'].size))
		
	return d
def load_survey(survey,metalkey='feh',selrc=False):

	import dtools
	
	d=autil.read_astro_catalog(survey,getdirec('phddir'))
	if survey.startswith('apogee'):
		d['tmasswise_j']=d['j']
		d['tmasswise_h']=d['h']
		d['tmasswise_ks']=d['k']
		d['teff']=d['param'][:,0]
		d['grav']=d['param'][:,1]
		d['feh']=d['fe_h']                                              # changed by Shourya April 10 2018
		d['m_h']=d['param'][:,3]
		d['vr']=d['vhelio_avg']
		d['field_id']=d['location_id']
		d1={'field_id':d['location_id']}
		s=set(d.keys())
		for key in ['pmra_ucac5','pmra_hsoy','pmra','pmdec_ucac5','pmdec_hsoy','pmdec','teff','grav','feh','vr','tmasswise_j','tmasswise_h','tmasswise_ks','ra','dec','glon','glat','m_h','apogee_id']:
			if key in s:
				ind=np.where(d[key]<-1e3)[0]
				d[key][ind]=np.nan
				d1[key]=d[key]
		if survey.startswith('apogee_rc'):
			d1['rad']=d['rc_dist']
		# bovy data
		for key in ['rc_galphi','rc_galz','rc_galr','vhelio_avg','metals','rc_dist']:
			if key in d.keys():
				d1[key]=d[key]	
		d=d1

	elif survey == 'galah':	
		
		use_sven = True
		
		if use_sven:
			print('using file from Sven (December 22, 2018)')
			
			d['teff']=d['teff']
			d['grav']=d['logg']
			d['feh']=d['fe_h']
			d['vr']=d['rv_guess']
			d['tmasswise_j']=d['j_m']
			d['tmasswise_h']=d['h_m']
			d['tmasswise_ks']=d['ks_m']     
			d['alpha_fe_cannon'] = np.zeros(d['teff'].size)
			   
		else:
			d['teff']=d['teff_cannon']
			d['grav']=d['logg_cannon']
			d['feh']=d['feh_cannon']
			d['vr']=d['rv_guess']
			d['tmasswise_j']=d['jmag']
			d['tmasswise_h']=d['hmag']
			d['tmasswise_ks']=d['kmag']        

		d['m_h']  = dtools.m_h_from_feh(feh=d['feh'],a_feh=d['alpha_fe_cannon'])		
		
		d1={'field_id':d['field_id']}
		s=set(d.keys())	
		#print(''); print('filtering out pilot'); d=fltr_pilot_galah(d,keep_=['','k2','tess'])	
		kylist=['parallax'
		        ,'parallax_error' 
		        ,'pmra'
		        ,'pmdec'
		        ,'pmra_ucac5'
		        ,'pmra_hsoy'
		        ,'pmdec_ucac5'
		        ,'pmdec_hsoy'
		        ,'teff'
		        ,'grav'
		        ,'feh'
		        ,'vr'
		        ,'tmasswise_j'
		        ,'tmasswise_h'
		        ,'tmasswise_ks'
		        ,'ra'
		        ,'dec'
		        ,'glon'
		        ,'glat'
		        ,'m_h'
		        ,'sobject_id'
		        ,'alpha_fe_cannon']
		for key in kylist:
			if key in d.keys():
				d1[key]=d[key]
		d=d1
		d['vmag_jk']=autil.jk2vmag(d['tmasswise_j'],d['tmasswise_ks'])
		tab.where(d,(d['vmag_jk']<14.0)&(d['vmag_jk']>9.0)&(np.isfinite(d['teff']))&(np.isfinite(d['grav']))
		             &(np.isfinite(d['feh']))&(np.isfinite(d['vr']))&(d['field_id']>=0)&(d['field_id']<7339))
		
	d['vmag_jk']=autil.jk2vmag(d['tmasswise_j'],d['tmasswise_ks'])	
	
	if ('pmra_hsoy' in s)and('pmra_ucac5' in s):
		d['mura']=(d['pmra_ucac5']+d['pmra_hsoy'])*0.5*1e-3
		d['mudec']=(d['pmdec_ucac5']+d['pmdec_hsoy'])*0.5*1e-3
		
		d['mura_ucac5'] = (d['pmra_ucac5'])*1e-3
		d['mura_hsoy'] = (d['pmra_hsoy'])*1e-3
		
		d['mudec_ucac5'] = (d['pmdec_ucac5'])*1e-3
		d['mudec_hsoy'] = (d['pmdec_hsoy'])*1e-3
	else:
		d['mura']=d['pmra']*1e-3
		d['mudec']=d['pmdec']*1e-3
	
	if selrc:
		cond = dtools.selfunc_rc(d['teff'],d['grav'],d[metalkey])
		tab.where(d,cond)
		d['dist']=dtools.distance_rc(d['tmasswise_j'],d['tmasswise_ks'],d['teff'],d['grav'],d[metalkey]) #feh			
		d['rad']=d['dist']
	
	print (d['ra'].size)
	return d
def load_galah_apogee_gaia(survey='all',selrc=False,quasar_offset=True):
		
	def get_galah():	
		
		d1=load_survey('galah',metalkey='m_h',selrc=selrc)		
		
		## cmatch with gaia cross-matched file?
		##fl = tabpy.read('/home/shourya/Documents/phd_work/GalahWork/sobject_iraf_53_gaia.fits')
		##tabpy.sortby(fl,'angular_distance')
		##tabpy.unique(fl,'source_id')	
		##tabpy.ljoin(d1,fl,'sobject_id','sobject_id')
		
		
		
		
		
		if 'mura' not in d1.keys():
			d1['mura'] = np.zeros(d1['ra'].size) -9999.
			d1['mudec'] = np.zeros(d1['ra'].size) -9999.		
		return d1
	
	def get_apogee():
		d3=load_survey('apogee_dr14',selrc=selrc)
		fl = tabpy.read('/home/shourya/Documents/phd_work/apogee_14/apogee_dr14_gaia.fits')
		tabpy.sortby(fl,'angdist')
		tabpy.unique(fl,'apogee_id')	

		# removes all multishaped keys to allow ljoin (check if you can resolve this later) # 091018
		for ky in fl.keys():
			if np.array(fl[ky].shape).size >1:
				del fl[ky]

		tabpy.ljoin(d3,fl,'apogee_id','apogee_id')
		
		if 'mura' not in d3.keys():
			d3['mura'] = np.zeros(d3['apogee_id'].size) -9999.
			d3['mudec'] = np.zeros(d3['apogee_id'].size) -9999.
		
		return d3

	def get_apogee_rc():
		d3=load_survey('apogee_rc_dr14',selrc=selrc)
		fl = tabpy.read('/home/shourya/Documents/phd_work/gaia/data/apogee_gaia.fits')
		tabpy.sortby(fl,'angdist')
		tabpy.unique(fl,'apogee_id')	

		# removes all multishaped keys to allow ljoin (check if you can resolve this later) # 091018		
		for ky in fl.keys():
			if np.array(fl[ky].shape).size >1:
				del fl[ky]
		
		tabpy.ljoin(d3,fl,'apogee_id','apogee_id')
		
		if 'mura' not in d3.keys():
			d3['mura'] = np.zeros(d3['apogee_id'].size) -9999.
			d3['mudec'] = np.zeros(d3['apogee_id'].size) -9999.
					
		return d3
	
	if survey =='all':
		print('')

		if selrc:
			print('combining APOGEE & GALAH RC....')
		else:		
			print('combining APOGEE & GALAH .....')		
		d1 = get_galah()
		d3 = get_apogee()
		####keep common keys only
		d1keys = np.array(d1.keys())
		d3keys = np.array(d3.keys())
		indl,indr,indnm = tabpy.crossmatch(d3keys,d1keys)
		for key in d1.keys():
			if key not in d1keys[indr]:		
				del d1[key]		
		tab.union(d1,d3,None)
		
	elif survey == 'galah':
		print('')
		
		if selrc:
			print('getting GALAH RC....')		
		else:
			print('getting GALAH ....')					
		d1 = get_galah()
	elif survey == 'apogee':
		print('')

		if selrc:
			print('selecting RC from APOGEE....')		
		else:
			print('getting APOGEE ....')					
		d1 = get_apogee()
		
	elif survey == 'apogee_rc':
		print('')

		if selrc:
			print('selecting RC from APOGEE-RC ....')		
		else:
			print('getting APOGEE-RC ....')					
		d1 = get_apogee_rc()

	print('')
	print('Not making parallax cuts!!!!!!!!!!!!!!!!!!!!')

	#p_errorcut = 0.2
	
	#tabpy.where(d1,condition= (np.isfinite(d1['parallax'])&(d1['parallax']>0.) &((d1['parallax_error']/d1['parallax'])<p_errorcut)))	
	

	if quasar_offset:
		print('')
		print('Applying quasar offset of +0.029 mas')
		d1['parallax']  = d1['parallax'] + 0.029
	
	
	d1['dist_gaia'] = 1.0/d1['parallax']
	
	# copy UCAC5 & HSOY averaged proper motion
	d1['mura_avg'] = d1['mura'].copy()
	d1['mudec_avg'] = d1['mudec'].copy()
	
	d1['mura'] = d1['pmra']*1e-03
	d1['mudec'] = d1['pmdec']*1e-03
		
	return d1
def load_galaxia_gaia(survey='',data=None):	
	flname = 'gaia_warp'
	loc = getdirec('home')+'/GalaxiaData'
	d = ebf.read(loc+'/'+flname+'.ebf')
	#d = mk_mini(d,fac=0.01,key='px')
	#sutil.ebfwrite(d,'tst',getdirec('desktop'))
	#d= ebf.read(getdirec('desktop')+'/tst.ebf')
	
	d['teff']=np.power(10.0,d['teff'])
	d['age_gyr']=np.power(10.0,d['age']-9.0)
	
	# copy absolute magnitudes
	d['tmass_j1']=d['tmass_j'].copy()
	d['tmass_h1']=d['tmass_h'].copy()
	d['tmass_ks1']=d['tmass_ks'].copy()
	d['vmag_jk1']=autil.jk2vmag(d['tmass_j'],d['tmass_ks'])
	
	d['tycho_bt1'] = d['tycho_bt'].copy()
	d['tycho_vt1'] = d['tycho_vt'].copy()
	
	d['gaia_g1'] = d['gaia_g'].copy()
	d['gaia_gbp'] = d['gaia_gbp'].copy()
	d['gaia_grp'] = d['gaia_grp'].copy()
	
	
	gutil.abs2app(d,isoch_loc=loc)
	
	## for new model can be added at anytime
	add_kinematics(d,thick=False)
	gutil.append_muvr(d)
	
	d['pmra_err']=np.zeros_like(d['glon'])+2.0
	d['pmdec_err']=np.zeros_like(d['glon'])+2.0
	d['mura']=np.random.normal(size=d['glon'].size,scale=d['pmra_err']*1e-3,loc=d['mura'])
	d['mudec']=np.random.normal(size=d['glon'].size,scale=d['pmdec_err']*1e-3,loc=d['mudec'])


	#if survey == 'gaia':
		#nside = 16
		#indx = autil.lb2hpix(nside,dg['glon'],dg['glat'])
		#l,b = autil.hpix2lb(nside,indx)
	
	
		#tab.where(d,(d['gaia_g']>2.0)&(d['gaia_g']<17.0))
		#tab.where(data,(data['phot_g_mean_mag']>2.0)&(data['phot_g_mean_mag']<17.0))
		#h=sutil.hist_nd([data['glon'],data['glat'],data['phot_g_mean_mag']],range=[[0,360.],[-90.0,90.],[2.0,17.0]],bins=[np.max(data['field_id'])+1,5])
		##h.data=h.data*4
		#ind=h.resample([d['field_id'],d['vmag_jk']])
		#tab.select(d,ind)
	#elif survey == 'apogee':
##		np.random.seed(12)
##		print('seed =12')
		#d['field_id']=d['location_id']
		#del d['location_id']
		#tab.where(d,(d['tmasswise_h']>7.0)&(d['tmasswise_h']<13.8)&((d['tmasswise_j']-d['tmasswise_ks'])>0.5))
		#tab.where(data,(data['tmasswise_h']>7.0)&(data['tmasswise_h']<13.8)&((data['tmasswise_j']-data['tmasswise_ks'])>0.5))
		#h=sutil.hist_nd([data['field_id'],data['tmasswise_h']],range=[[0,np.max(data['field_id'])+1],[7.0,13.8]],bins=[np.max(data['field_id'])+1,34])
		##h.data=h.data*4
		#ind=h.resample([d['field_id'],d['tmasswise_h']],fsample=1.0)
		#tab.select(d,ind)

	#else:
##		tab.where(d,(d['tmasswise_h']>7.0)&(d['tmasswise_h']<15.0)&(d['vmag_jk']<15.0))
##		tab.where(d,(d['tmasswise_h']>7.0)&(d['tmasswise_h']<13.8))
		#tab.where(d,(d['tmasswise_h']>7.0))
		#print d['ra'].size
	
	return d
def fltr_pilot_galah(mydata,keep_=['','k2','tess']):  

	print('Removing PILOT [cobid>1403010000] & -ve fid only')
	tabpy.where(mydata,condition=(mydata['field_id'] >-1)&(mydata['cob_id']>1403010000)) 			
	
	print('')
	print('Using only: ')
	print(keep_)				

	tabpy.where(mydata,condition=(mydata['progname']==keep_[0])|(mydata['progname']==keep_[1])|(mydata['progname']==keep_[2]))
		
	return mydata
def dgaia(ver='dr2'):	
	
	if ver == 'dr2':
		fname = 'gaia_mini'
		loc = getdirec('phddir')+'/gaia/data'					
	elif ver == 'edr3':
		loc = getdirec('pdocdir')+'/science/edr3/gedr3data'					
		
		fname = 'rvs_poege5_ext'
	# elif ver == 'dr3':
		# loc = getdirec('pdocdir')+'/science/edr3/gedr3data'					
		
		# fname = 'rvs_poege5_ext'
	
	print(loc)	
	print('reading.... '+ver+'...'+fname)
	d1=ebf.read(loc+'/'+fname+'.ebf')		
	if ver == 'edr3':
		print('')
		d1_feh = 	ebf.read(loc+'/'+fname+'_feh.ebf')
		tabpy.ljoin(d1,d1_feh,'source_id','source_id')
	# d1['lgc'] = d1['glongc'].copy()

	return d1
def dgalah(create=False):
	loc = getdirec('phddir')+'/GalahWork'		
	if create:			
		dt = load_galah_apogee_gaia(survey='galah')
		getkin(dt,'dist_gaia',dtype='data')
		ebfwrite(dt,'galah_use',loc)	
		d = dt.copy()	
	else:		
		dt=ebf.read(loc+'/'+'galah_use_dr2.ebf')		
		
		dt2=ebf.read(loc+'/'+'galah_use.ebf')	
		#dsven = tabpy.read(loc+'/GALAH_iDR3_v1_181221.fits')
		dsven = tabpy.read(loc+'/GALAH_iDR3_main_alpha_190529.fits')
		
		indl,indr,indnm=tabpy.crossmatch(dt['sobject_id'],dsven['sobject_id'])

		dt['feh_new'] = np.zeros(dt['feh'].size) + np.nan
		dt['feh_new'][indl] = dsven['fe_h'][indr]	
		
		dt['alpha_new'] = np.zeros(dt['alpha_fe_cannon'].size) + np.nan
		dt['alpha_new'][indl] = dsven['alpha_fe'][indr]
		
		d = dt.copy()
		d['lgc'] = d['glongc'].copy()
		
	return d
def dapogee(filetyp=''):
	'''
	do this properly for DR17
	'''
	gloc = '/net/gaia2/data/users/gaia'
	loc = gloc+'/apogee_dr17'
	# # loc = gloc+'/apogee_16'
	
	if filetyp == 'rc':
		print('reading apogee-dr17 rc file')		
		loc = gloc+'/apogee_dr17'		
		dt = tabpy.read(loc+'/apogee-rc-DR17.fits')
		
		# correctly format apogee_id in the RC catalog
		dt['apogee_id'] = dt['apogee_id'].astype(str)
		tmpval = []
		for val in dt['apogee_id']:
				tmpval.append(val.replace(" ","")) 
		dt['apogee_id'] = np.array(tmpval)
		
			
	else:
		# print('reading apogee-dr16')
		# dt = tabpy.read(loc+'/allStar-r12-l33.fits') 		
		print('reading apogee-dr17')
		dt = tabpy.read(loc+'/allStar-dr17-synspec_rev1.fits') 		
		dt['apogee_id'] = dt['apogee_id'].astype(str)
		
		
	return dt
	
	
	


class dtoy_sanjib(object):




	def __init__(self,name=None,readsnaps=False):		
		self.loc = getdirec('phddir')+'/toy_phasemix'
		
		if name is not None:				
			self.name = name
			self.filename = name+'.ebf'			
		else:	
			self.filename = 'ridge_galpy2.ebf'			
		
		print(self.filename)
		
		if readsnaps == False:
			self.mkfile()				
			



	def mkfile(self):			
		self.data=ebf.read(self.loc+'/'+self.filename,'/data/')	
		self.siminfo=ebf.read(self.loc+'/'+self.filename,'/siminfo/')

		self.rscale=self.siminfo['rscale']
		self.vscale=self.siminfo['vscale']				
		self.npoints = self.data[list(self.data.keys())[0]][0,:].size


	def getsnaps(self,snapnum=99):
		'''
		Only works in readsnaps=False mode
		
		'''
		print('reading saved snapshots')
		print(snapnum)
		loc = self.loc+'/'+self.name+'_snapshots'
		self.d = tabpy.read(loc+'/'+str(snapnum)+'.ebf')
		
	
		return self.d		
	def gettscale(self):
		'''
		Only works in readsnaps=False mode (check this!)
		
		'''
		print('reading tscale from saved snapshots')
		loc = self.loc+'/'+self.name+'_snapshots'
		timescale = tabpy.read(loc+'/timescale.ebf')
		
	
		return timescale['ts']		
			
	
		
	def getrun(self,i=70,**kwargs):

		endpoint = self.npoints		
		if 'points' in kwargs.keys():			
			endpoint = kwargs['points']
			print(endpoint)

		d = {}
		for key in self.data.keys():
			d[key] = self.data[key][i,:endpoint].copy()
			

		print('scaling velocity and distance')
		for key in d.keys():
			if key.startswith('v'):
				d[key] = d[key]*self.vscale
			elif key.startswith('r'):
				d[key] = d[key]*self.rscale			
		
		# assign spiral-arm ids:		
		d['id']=(np.arange(d['rgc'].size)%(d['rgc'].size/16))/(d['rgc'].size/(16*4))				
		
		self.d = d.copy()
		
		return self.d


# Distance catalogues
def add_pauld(d):
	'''
	McMillan distances are in parsec!
	'''
	flloc = getdirec('phddir')+'/gaia/data'
	fl=tabpy.read(flloc+'/GaiaDR2_RV_star_distance_paul.ebf')
	ind=np.where((fl['distance_error']/fl['distance'])>0.2)[0]
	fl['distance'][ind]= np.nan
	fl['distance'] = fl['distance']*1e-03
	fl['distance_error'] = fl['distance_error']*1e-03
	tabpy.ljoin(d,fl,'source_id','source_id')
def mk_schon_dist():
	desktop = dtools.getdirec('desktop')	
	
	timeit = dtools.stopwatch()
	timeit.begin()	
	
	#subdir = 'modoffset'; flname = 'gaiaRVdelpeqspdelsp43'
	subdir = '54offset'; flname = 'gaiaRVdelp54delsp43'
	
	loc = dtools.getdirec('phddir')+'/gaia/data/schonrich_d/'+subdir
	d1 = ascii.read(loc+'/'+flname+'.txt')
	
	dt = {}
	dt['source_id'] = d1['sourceid'].copy()
	dt['dschon'] = d1['E_dist'].copy()
	dt['parallax_schon'] = d1['parallax'].copy()
	dt['parallax_parallaxerr_schon'] = d1['parallax/parallaxerr'].copy()
	dtools.ebfwrite(dt,subdir+'_use',loc)
	
	
	timeit.end()	

		
# stats
def midx(x):
	midx = (x[1:]+x[0:-1])*0.5	
	return midx
def gaus(x,x0,sigma):
	 return (1.0/(np.sqrt((2*np.pi))*sigma))*np.exp(-0.5*(np.power((x-x0)/sigma,2.0)))	
def boots(data,iters=2):
	from astropy.stats import bootstrap
	from astropy.utils import NumpyRNGContext	

	test_statistic = lambda x: (np.mean(x))
	
	with NumpyRNGContext(1):
		bootresult = bootstrap(data, iters,bootfunc = test_statistic)	
		
		
	boot_std = np.std(bootresult)		
	return boot_std, bootresult
	
	
def bootstrap_(data,iters=2,bootfunc=None):
	from astropy.stats import bootstrap
	from astropy.utils import NumpyRNGContext	
	
	with NumpyRNGContext(1):
		bootresult = bootstrap(data, iters,bootfunc=bootfunc)	
			
	
	
	return	bootresult
def mystat(x,uplow=False):#-----------bmcmc type print
	temp=np.percentile(x,[16.0,84.0,50.0])
	sig=(temp[1]-temp[0])/2.0
	xmean=np.mean(x)
	d2=np.floor(np.log10(sig))
	return np.round(xmean*(10**(-d2)))/10.0**(-d2)
def get_bmcmc2():
	'''
	to import bmcmc2 from external directory
	01/Nov/2018
	
	'''
	import sys	
	loc = getdirec('pdocdir')+'/py_scripts'
	
	sys.path.insert(0,loc)
	
	import bmcmc2 as bmcmc
	
	return bmcmc	
	
def pdf2cdf(pdf_,xv=None,plotit=False,getlims=False,usesig=1,sample_=False,nsamples=10,interp_=False,yv=0):
	'''
	xv = xpoints 
	pdf = evaluated at xv
	assuming that xv is already ordered
	
	'''
	
	cdf = np.cumsum(pdf_)
	cdfval = cdf/cdf[-1]
	if plotit:
		plt.plot(xv,cdfval)
		
	if getlims:
		zm = interp1d(cdfval,xv,bounds_error=False,fill_value=(cdfval[0],cdfval[-1]))	
		
		dlta = stats.chi2.cdf(usesig**2.,1)/2.
		xmin, xmax = zm(0.5 - dlta),zm(0.5+dlta)
		
		return cdfval, xmin, xmax
	
	if sample_:
		zm = interp1d(cdfval,xv,bounds_error=False,fill_value=(cdfval[0],cdfval[-1]))
		
		cdfpnts = np.random.uniform(size=nsamples)
		xpnts = zm(cdfpnts)
		
		return xpnts
	if interp_:

		zm = interp1d(xv,cdfval,bounds_error=False,fill_value=(xv[0],xv[-1]))
		
		ypnts = zm(yv)
		
		return ypnts
	
		
	else:		
		return 	cdfval
		
	
def get_pyia():
	'''
	to import pyia from external directory
	06/Nov/2019
	
	'''
	import sys	
	#loc = '/home/shourya/Documents/phd_work'
	loc = getdirec('phddir')+'/pyia-master'
	
	sys.path.insert(0,loc)
	import pyia 
	
	return pyia

def chisq(data,model):
	
	chisq = np.square(data - model)/model		
	return chisq
def chi2(x1,x2,bins=10,range=None):
    h1,be1=np.histogram(x1,range=range,bins=bins,density=False)
    h2,be2=np.histogram(x2,range=range,bins=bins,density=False)
    temp=np.sum(h1,dtype=np.float64)/np.sum(h2,dtype=np.float64)
    err=h1+h2*np.square(temp)
    ind=np.where((h1+h2)>0)[0]
    return np.sum(np.square(h1[ind]-h2[ind]*temp)/err[ind],dtype=np.float64)#/(ind.size-1)


def chunkwriter(i):
	mydata = mydatacopy.copy()
	idkey = idkey_
	indl, indr, indnm = tabpy.crossmatch(chunk_list[i],mydata[idkey])
	d = {}
	for ky in mydata.keys():
		d[ky] = mydata[ky][indr]	
	if write_as_=='fits':
		fitswrite(d,'data_part_'+str(i),loc_)	
	elif write_as_ == 'ebf':
		ebfwrite(d,'data_part_'+str(i),loc_)		
	return

def mk_chunks(mydata,chunk_size=1000,loc='',refkey='glon',idkey='fid',write_as='ebf',ncpu=5):
	
	'''
	clean it right away!!
	'''
	

	global mydatacopy
	global loc_
	global write_as_
	
	global chunk_list
	global idkey_
	
	if idkey in mydata.keys():
		raise RuntimeError(idkey+' already present')
		
	if chunk_size > mydata[refkey].size:
				raise Exception('Reduce chunk size (bigger than data size)')
	
	mydata[idkey]=np.array([i for i in range(mydata[refkey].size)])
	items, chunk = mydata[idkey], chunk_size
	# chunk_list = np.array(zip(*[iter(items)]*chunk))	#python2
	chunk_list = np.array(list(zip(*[iter(items)]*chunk))) #python3



	mydatacopy = mydata.copy()	
	loc_ = loc
	write_as_ = write_as
	idkey_ = idkey
	
	####		
	
	# # # for i in range(len(chunk_list)):
		# # # indl, indr, indnm = tabpy.crossmatch(chunk_list[i],mydata[idkey])
		# # # d = {}
		# # # for ky in mydata.keys():
			# # # d[ky] = mydata[ky][indr]	
		# # # if write_as=='fits':
			# # # fitswrite(d,'data_part_'+str(i),loc)	
		# # # elif write_as == 'ebf':
			# # # ebfwrite(d,'data_part_'+str(i),loc)	
	

	
	print('running pool')
	
	i = np.arange(len(chunk_list)) 
	ifinal = i[-1]+1
	from multiprocessing import Pool
	p=Pool(ncpu)
	p.map(chunkwriter,i)

	print('end pool')
			
	####	
	
	print(str(ifinal)+' files written')
	
	chunk_list_flt = chunk_list.flatten()
	indl, indr, indnm = tabpy.crossmatch(chunk_list_flt,mydata[idkey])
	
	if indnm.size >0:
		d = {}
		for ky in mydata.keys():
			d[ky] = mydata[ky][indnm]	
			
		if write_as=='fits':			
			fitswrite(d,'data_part_'+str(ifinal),loc)	
			
		elif write_as == 'ebf':
			ebfwrite(d,'data_part_'+str(ifinal),loc)	

		
		print('')
		print('Remainder file contains = '+str(d[refkey].size))
	

	return 


	
def cmb_chunks(loc='',svloc='',refkey='',write_as='ebf',read_format='.fits',svname='combined'):
	
	
	import natsort
	from natsort import natsorted, ns	
	
	timeit = stopwatch()
	timeit.begin()	


	import os
	if svloc == '':
		svloc = os.getcwd()
	
	files = os.listdir(loc)
	files = natsorted(files)
	files = np.array(files)
	

	
	####-- shuffle - currently running! (delete this line upon completion)
	#print('Total = '+str(len(files)))
	#inds = np.arange(0,len(files),1)
	#np.random.shuffle(inds)
	#ind = inds[:100]
	#files = files[ind]
	# files = files[:10]
	####-----------------
	
	file_list = []
	for fl in files:	
		if fl.endswith(read_format):  
			file_list.append(fl)
	
	file_list = np.array(file_list)				

	#print(len(file_list))
	print(file_list)		

						
	d1 = tabpy.read(loc+'/'+file_list[0])
	

	for inum, fl in enumerate(file_list[1:]):
		print(str(inum)+'......')
		print(fl)
		d2 = tabpy.read(loc+'/'+fl)
		tabpy.union(d1,d2,None)
		#print(d1[refkey].size)
		
	if refkey != '':
		tabpy.sortby(d1,refkey)
		tabpy.unique(d1,refkey)
		#print(d1[refkey].size)		
		
		
	if write_as=='fits':
		dt = Table(d1)
		dt.write(svloc+'/'+svname+'.fits',format='fits',overwrite=True)

	elif write_as == 'ebf':
		ebfwrite(d1,svname,svloc)	

	
	print('')
	print('combined file contains = '+str(d1[list(d1.keys())[0]].size))
	
	timeit.end()	
	return 
	
def cmb_chunks_keys(loc='',svloc='',refkey='glon',write_as='ebf',read_format='.ebf',svname='combined',keys='',pky=False,nosavrefky=False):
	'''
	February 23 2019
	To combine chunks based on a given keylist
	! only works if read_format = ebf
	'''

	import os
	import natsort
	from natsort import natsorted, ns	
	
	if svloc == '':
		svloc = os.getcwd()
	
	files = os.listdir(loc)
	files = natsorted(files)
	files = np.array(files)
	
	file_list = []
	for fl in files:	
		if fl.endswith(read_format):  
			file_list.append(fl)
	
	file_list = np.array(file_list)

	if read_format == '.ebf':
		d1 = ebfread_keys(loc+'/'+file_list[0],keys=keys)

		for inum, fl in enumerate(file_list[1:]):
			print(str(inum)+'......')
			d2 = ebfread_keys(loc+'/'+fl,keys=keys)				
			tabpy.union(d1,d2,None)


	tabpy.sortby(d1,refkey)
	tabpy.unique(d1,refkey)


	ds = {}
	for ky in d1.keys():
		ds[str(ky)] = d1[ky].copy()
		
	
	## added on December 14, 2022 [INAF TORINO]
	if nosavrefky:
		del ds[refkey]
		
	if write_as=='fits':
		dt = Table(ds)
		dt.write(svloc+'/'+svname+'.fits',format='fits',overwrite=True)

	elif write_as == 'ebf':
		ebfwrite(ds,svname,svloc)	

	
	print('nosavrefky is ..='+str(nosavrefky))
	print(ds)
	
	print('')
	print('combined file contains = '+str(ds[list(ds.keys())[0]].size))
	
	
	return 

def scott_bin(N,sigma,dim=1):
	'''	  	
    		
    NAME:

    PURPOSE:
    	calculates bin width using Scott's rule     
    INPUT:
		N = total number of data points
		sigma = standard dev along a dimension
		dim (=1 by default) = number of dimensions

    OUTPUT: bin width
       
    HISTORY: generalised formula (16 Aug, 2023)
        
    https://ui.adsabs.harvard.edu/abs/1992mde..book.....S/abstract    
    
    https://stats.stackexchange.com/questions/114490/optimal-bin-width-for-two-dimensional-histogram	
	https://arxiv.org/pdf/physics/0605197.pdf
	'''	
	
	
	b_width = (3.5*sigma)/(N**(1./(2. + dim)))
	
	# b_width = (3.5*sigma)/(N**(1./3.))
	# b_width = (2.15*sigma)/(N**(1./5.))
	
	return b_width




def rms(data):
	
	return np.sqrt(np.mean(data**2.))
def entropy(prob):
	'''	  	
    		
    NAME:

    PURPOSE:
    	calculates entropy     
    INPUT:
		probability

    OUTPUT:
       
    HISTORY:
        
	
	'''
	lnprob = np.log(prob)	
	ind_en = indfinite(lnprob)
	ent = -np.sum(prob[ind_en]*lnprob[ind_en])

	return ent

def modulus(x,y,z):
	'''
	returns the modulus of the x and y and z components
	'''	
	
	val = np.sqrt(x**2 + y**2 + z**2)
	return val


def fitgauss(plotchars,xval,yval,amp=5, cen=5, width=1):

	'''
    NAME:

    PURPOSE:
    	fits a Gaussian profile to a given data curve (x,y)
    INPUT:
		x,y

    OUTPUT:
       
    HISTORY: January 24, 2022 (RUG)
        	
	'''

	from numpy import exp, loadtxt, pi, sqrt
	from lmfit import Model
	

	def gaussian_func(x, amp, cen, wid):
	    """1-d gaussian: gaussian(x, amp, cen, wid)"""
	    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))	

	
	gmodel = Model(gaussian_func)
	result = gmodel.fit(yval, x=xval, amp=amp, cen=cen, wid=width)
	
	
	# # print(result.fit_report())
	
	
	if plotchars['plot_']:		

		plt.close('all')
		
		# plm=putil.Plm2(1,2,xsize=8.0,ysize=3.,xmulti=False,ymulti=False,full=True,slabelx=0.7,slabely=0.07)			
		plm=putil.Plm1(1,1,xsize=8.0,ysize=6.,xmulti=False,ymulti=False,full=False,slabelx=0.7,slabely=0.07)			
		plm.next()
		plt.plot(xval, yval, 'k.') 
		plt.plot(xval, result.best_fit, '-', label='best fit') 
		plt.plot(xval, result.init_fit, '--', label='initial fit')		; 		plt.legend()
		plt.axvline(result.best_values['cen'])			
		
		
		plt.title(plotchars['title'])
		plt.text(0.5,0.95,str(np.round(result.best_values['cen'],3))+', '+str(np.round(result.best_values['wid'],3)),transform=plt.gca().transAxes,color='orange',fontsize=10)	
	
		if plotchars is not None:
			
			# plm.next()
			# plt.plot(plotchars['x2'],yval,'r.')
			
			plm.tight_layout()
	
			if plotchars['plotsave']:			
				plt.savefig(plotchars['plotloc']+'/fitgauss_'+plotchars['plotname']+'.png')
	
	return result.best_values['cen'],result.best_values['wid'],result


def get_kde(data,kernel='gaussian',use_scott=False,bandwidth=1,showleg=False,density=True,outbins=1000,lwdth = 2,alpha=0.5,typ='',prnt=True):	
	
	'''
		
    NAME: get_kde

    PURPOSE: to provide a kernel density fit


    INPUT: data (1d array)
		   kernel: 'gaussian' or 'tophat' (check for updates to see if other kernels are available)
		   use_scott: True/False [to plot data histogram using Scott's binning]
					  if False, bins is computed using data percentiles
		   bandwidth: used for kde estimation [same units as data]
		   outbins: N bins to be used for output data
			

    OUTPUT: a plot of data,
			X [x axis where kde is evaluated]
			Y [kde evaluated at X]
    HISTORY: September 28, 2021 (Groningen)       
			 April 07, 2023 (INAF-Torino)     [cleaned up to allow kde sampling at user-defined bins]  
	
	https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#sphx-glr-auto-examples-neighbors-plot-kde-1d-py
	
	'''		


	

	from sklearn.neighbors import KernelDensity

	xmin,xmax = np.nanpercentile(data,1),np.nanpercentile(data,99)
	fac = 0.25
	if use_scott:
		if prnt:
			print('using Scotts rule')
		bandwidth = fac*scott_bin(data.size,np.std(data)) 
		# print('Scott bin width = '+str(bn_width))
		bins = 	int((xmax-xmin)/bandwidth)
		
	else: 					
		bins = int((xmax - xmin)/bandwidth)	
	
	if prnt:	
		print(bandwidth,bins)
		
		
		print('using '+kernel+' kernel')
	
	X = data[:,np.newaxis] 	
	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)

	X_plot = np.linspace(xmin, xmax, outbins)[:, np.newaxis]		
	log_dens = kde.score_samples(X_plot)
	
	

	if typ =='':
		plt.hist(data,bins=bins,density=density,range=[xmin,xmax],label='data',histtype='step',alpha=alpha)
	plt.plot(X_plot[:, 0], np.exp(log_dens),label='fit',linewidth=lwdth,color='orange')

		
			
	##########	
	
	if showleg:
		plt.legend()	
	

	return X_plot[:, 0], np.exp(log_dens)



def getlike_basic(model_,data_,sigval=1):
	'''
	very basic log likelihood 
	'''
	

	tmp1 = data_ - model_ 
	sig1 = sigval
	
	# indf = dtools.indfinite(1./sig1,print_=False)	
	# sig = sig1[indf]
	# tmp = tmp1[indf]
	
	sig = sig1
	tmp = tmp1
	
	loglike_ = -0.5*(np.sum((((tmp)/sig)**2) - np.log(2.*np.pi*sig)))
	minloglike_ = -loglike_
	return minloglike_, sig
	
		
	
def getlikelihood(model,data,noise=[],useall=False,typ='gauss'):
	
	'''
	
	NAME: likelihood_poiss
	
	PURPOSE:
		returns Poissonian likelihood
	INPUT:
		model
		data
		noise (if typ == 'poiss')
		
		typ: 'gauss' 'poiss'
	
	OUTPUT:
	
	HISTORY: February 12, 2024 (INAF-OATO)	
	
	
	returns: log L 
	
	useall = uses all three terms in the log l
	'''
	
	
	if typ =='poiss':
	
		tmp1 = - model
		tmp2 = data*np.log(model)
		
		tmp = tmp1 + tmp2 
		if useall:	
			tmp3 = - np.log(scipy.special.factorial(data))
			tmp += tmp3
		
		return tmp

	elif typ =='gauss':
		
		if len(noise) == 0:
			raise Exception('using Gaussian likelihood, provide noise')
			
		

		temp = 0	
		pdf_ = scipy.stats.norm.logpdf(data,loc=model,scale=noise)
		temp+= pdf_
		
		return pdf_
		
		



def eqn_linear(x,m,cy=None,cx=None):
	'''
	input: xpoints (array)
		   m (gradient)
		   cy (y-intercept, None by default)
		   cx (x-intercept, None by default); if provided, make sure to have cy=None
		   
	History: July 10, 2022 (INAF, Torino)
	'''
	
	if cy is not None:
		c = cy
	else:
		c = -m*cx

	y = (m*x) + c
	
	return y



def getpercentiles(data,useval=[1,16,50,84,99],prnt=True):
	
	'''
	gets percentiles at specified intervals. 
	default at [1,16,50,84,99]
	
	'''
	
	if prnt:
		print(useval)
	pval = []
	for val in useval:
		pval.append(np.nanpercentile(data,val))
	
	return np.array(pval)




# File reading
def ebfread_keys(file1,keys,printkeys=False):
	'''
	PURPOSE: reads ebf file (specified keys only)
	good for reading huge files
	'''
	
	data = {}
	for key in keys:
		if printkeys:
			print('')
			print(key)
			print('')
		data[key] = ebf.read(file1,'/'+key)
			
	return data

def readh5(file2,path=None):
	ds = astropy.io.misc.hdf5.read_table_hdf5(file2,path=path)
	
	return ds


def pickleread(file1):
	
	'''
	read pickle files
	input: fileloc+'/'+filename
	
	'''
	import pickle	
	data = pickle.load(open(file1,'rb'))	
	
	
	return data



# File writing
def ebfwrite(data,nam,loc,prnt=True):
	'''
		
	'''

	ebf.write(loc+'/'+nam+'.ebf','/',data,'w')	
	if prnt:
		
		print(nam+'.ebf written to '+loc)			
	return
def ebfappend(file1,data,ext):
	'''
	file1 : ebf file to be modified
	data: new key data being added
	ext: new key name being added
	'''	
	

	ebf.write(file1,'/'+ext,data,'a')	
	print(file1+' appended')			
	return
	
def fitswrite(data,nam,loc,prnt=True):
	
	dt = Table(data)
	dt.write(loc+'/'+nam+'.fits',format='fits',overwrite=True)	
	if prnt:
		print(nam+' .FITS written to '+loc)		
	return 

def picklewrite(data,nam,loc,prnt=True):
	'''
	write files using pickle
	'''
	
	
	
	import pickle
	
	pickle.dump(data,open(loc+'/'+nam+'.pkl','wb'))	
	if prnt:
		print(nam+' .pkl written to '+loc)	
		
	return 



def findkey(data,name,look=''):
	'''

	Fix runtime error issue (June 27 2018)

	last update: April 23 2018
	
	'''

	kyvals = []
	if look == '':
		for key in data.keys():
			if key.startswith(name):
				# print(key)
				kyvals.append(key)
			#else key.startswith(name)==False:
				#raise RuntimeError('key not found')
			
	if look=='any':
		for key in list(data.keys()):
			if name in key:
				#print(key)		
				kyvals.append(key)
	if len(kyvals) > 0:
		# print(str(len(kyvals))+' found')
		return kyvals					
	else:
		return None


def mk_textable():
	fldir = '/home/shokan/Downloads/'
	flnm = 'galaxy10myparameterfile'
	
	FILE=np.loadtxt(fldir+flnm+'.txt', dtype=str)#,usecols=(0,7,10))
	col=zip(*FILE)
	tab=Table(col)
	tab.write('table',format='latex')
	print(tab)	
	return

# plotting
def one2one(x,y,rng=[],alpha=0.2,s=0.2,**kwargs):
	
	lbl = ''
	lnstyle = '-'
	clr='black'
	if 'label' in kwargs.keys():
		lbl = kwargs['label']
	if 'linestyle' in kwargs.keys():
		lnstyle = kwargs['linestyle']
	if 'color' in kwargs.keys():
		clr = kwargs['color']


	# plt.plot(x,y,'-',color=clr,label=lbl)
	if len(rng) > 0.:
		xpnt = np.linspace(rng[0],rng[1],100)	
		plt.xlim([rng[0],rng[1]])
		plt.ylim([rng[0],rng[1]])
		xpnt = np.linspace(0.,10.,100)
		plt.plot(xpnt,xpnt,linestyle='--')		
		plt.plot(xpnt,xpnt,'.')		
		
		
	else:
		print('here..')
		xmin = np.nanpercentile(x,1)
		xmax = np.nanpercentile(x,99)
		
		ymin = np.nanpercentile(y,1)
		ymax = np.nanpercentile(y,99)
		
		print(xmin,xmax)
		xpnt = np.linspace(xmin,xmax,100)
		plt.plot(xpnt,xpnt,linestyle='--')		
		# plt.plot(xpnt,xpnt,'.')		
	
		
	plt.scatter(x,y,alpha=alpha,s=s,color=clr)

	#plt.legend()

	
	return

def png2movie(readdir,savdir,flname='movie',fmt='gif',duration=1.):
	'''
	Purpose: make a gif from set of images
	readdir = directory where set of images are
	savdir = directory where to save the final movie
	flname = filename
	
	#dtools.png2movie(desktop+'/snaps/',desktop)	
	
	'''
	
	from PIL import Image as image
	import imageio	
	import natsort
	from natsort import natsorted, ns
	
	images = [] 
	filenames = os.listdir(readdir)
	filenames = natsorted(filenames)
	filenames = np.array(filenames)
	
	fps = 1./duration
	
	for filename in filenames:
		filename = readdir+'/'+filename 
		images.append(imageio.imread(filename))
		
	if fmt == 'gif':	
		imageio.mimsave(savdir+'/'+flname+'.gif', images,duration=duration)	
	elif fmt == 'mp4':
		imageio.mimsave(savdir+'/'+flname+'.mp4', images,fps=fps)	


	#import images2gif
	#from images2gif import writeGif	
	#for filename in filenames:
		#filename = readdir+'/'+filename 
		#images.append(image.io.open(filename))
	#imageio.mimsave(savdir+'/'+flname+'.gif', images,duration=0.8)	
	###writeGif("images.gif",images,duration=0.3,dither=0)	
	
	
	return


def mkcirc(rval):
	
	theta = np.radians(np.linspace(0.,360.,100))
	xval = rval*np.cos(theta)
	yval = rval*np.sin(theta)
	
	return xval, yval

def plot3d(x,y,z,setlim=False,xmin=0,xmax=10,ymin=0,ymax=10,zmin=0,zmax=10,color='red',s=0.2,alpha=0.2):
	# def plot3d(x,y,z,indset1,indset2,setlim=False,xmin=0,xmax=10,ymin=0,ymax=10,zmin=0,zmax=10,color='red'):	
	from mpl_toolkits.mplot3d import Axes3D
	from mpl_toolkits.mplot3d.art3d import Poly3DCollection

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d') 
	# ax.scatter3D(x,y,z, marker='p',s=1,alpha=0.7, color=color)		
	ax.scatter3D(x,y,z, marker='p',s=s,alpha=alpha, color='red')		
	# ax.scatter3D(x,y[indset2],z[indset2], marker='p',s=1,alpha=0.7, color='blue')		
	
								
	ax.set_xlabel('X (kpc)')
	ax.set_ylabel('Y (kpc)')
	ax.set_zlabel('Z (kpc)') 		
	
	if setlim:
		
		ax.set_xlim([xmin,xmax])
		ax.set_ylim([ymin,ymax])
		ax.set_zlim([zmin,zmax])
	
	plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99) 
	
	
	
	return
	
def threed(x,y,z):
#---------- Plot 3D ------------------------#
	fig = plt.figure();	ax = fig.add_subplot(111, projection='3d') 
	ax.scatter(x, y,z, marker='p',s=1,alpha=0.7)						
	ax.set_xlabel('X (kpc)');	ax.set_ylabel('Y (kpc)'); ax.set_zlabel('Z (kpc)') 		
	
	return
	
def mkline(m=1.,c=1.,x=[0.,1.],color='black',linestyle='-'):
	
	xp = np.linspace(x[0],x[1],100)
	
	y = m*xp + c
	
	plt.plot(xp,y,color=color,linestyle=linestyle)
	
	return 


def invert(axis):
	
	if axis == 'x':
		plt.gca().invert_xaxis()
	elif axis =='y':		
		plt.gca().invert_yaxis()
	elif axis == 'both':
		plt.gca().invert_xaxis()		
		plt.gca().invert_yaxis()		
	return


def pltLabels(xlabel, ylabel, lblsz=None):
	if lblsz:
		plt.xlabel(xlabel, fontsize = lblsz)
		plt.ylabel(ylabel,fontsize = lblsz)
		
	else:
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)		
	
	return

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


def dens_logmeth(mode='vol',plotit=False,rlogmin=-2,rlogmax=3,nbins=128):

	'''
	pwelan logmeth
	'''
	bins = np.logspace(rlogmin, rlogmax, nbins)
	bin_cen = (bins[1:] + bins[:-1]) / 2.
	H,edges = np.histogram(dt['rgc'], bins=bins, weights=np.zeros_like(dt['rgc']) + 1/dt['rgc'].size)					
	
	if mode =='vol':
		V = 4/3.*np.pi*(bins[1:]**3 - bins[:-1]**3)		
		densval_ = (H / V)
	else:	
		A = np.pi*(bins[1:]**2 - bins[:-1]**2)		
		densval_ = (H / A)		
		
	if plotit:	
		plt.loglog(bin_cen, densval_, marker=None, label='Particles', color='red')

	return bin_cen,densval_
		
def cmaps(key=''):
	'''
	prints details about colorcet colorschemes
	'''
	for ky in cc.cm.keys():
		if ky.startswith(key):			
			print(ky)
	
	
	return			

def colbar(im,ax,label='',orientation='vertical',pad=0.05,ticks=None):
	'''
	
	ax = plt.gca()
	'''
	print('pad = ')	
	print(pad)	
	from mpl_toolkits.axes_grid1 import make_axes_locatable	
	from matplotlib.transforms import Bbox	

	divider1 = make_axes_locatable(ax)
	cax = divider1.append_axes("right", size="5%",pad=pad)			
	
	if ticks is not None:		
		plt.colorbar(im,cax,orientation=orientation,label=label,ticks=ticks)
	else:
		plt.colorbar(im,cax,orientation=orientation,label=label)

	return

def clevel(h1,levels=[1.,0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],colors='red',linestyles='dotted',label=None,coords=False):
	'''
	Must be normed=True!
	
	'''
	xbin_wdth = (h1.locations[0][1] - h1.locations[0][0])
	ybin_wdth = (h1.locations[1][1] - h1.locations[1][0])
	z = (h1.data.reshape(h1.bins))*xbin_wdth*ybin_wdth
	
	n = 1000
	t = np.linspace(0, z.max(), n)
	integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
	
	from scipy import interpolate
	
	f = interpolate.interp1d(integral,t,bounds_error=False,fill_value=(t[0],t[-1]))
	
	t_contours = f(np.array(levels))
	
	
	xe = np.linspace(h1.range[0,0],h1.range[0,1],h1.bins[0]+1)
	ye = np.linspace(h1.range[1,0],h1.range[1,1],h1.bins[1]+1)
	xc = (xe[0:-1]+xe[1:])/2.0
	yc = (ye[0:-1]+ye[1:])/2.0
	x=np.meshgrid(xc,yc,indexing='ij')
	
	cntour = plt.contour(x[0],x[1],z, t_contours,colors=colors,linestyles=linestyles)



	#################	
	if coords:
		continf = {}
		continf['rvals'] = []
		ncnts = len(cntour.collections)
		
		for j in range(ncnts):
			continf[str(j)+'x'] = []
			continf[str(j)+'y'] = []
		
			plens = len(cntour.collections[j].get_paths())
			rvals = []
			for i in range(plens):
				ps = cntour.collections[j].get_paths()[i].vertices
				rvals.append(np.max(np.sqrt(ps[:,0]**2. + ps[:,1]**2.)))	# using max seems to better trace the contour radius	
				continf[str(j)+'x'].append(ps[:,0])
				continf[str(j)+'y'].append(ps[:,1])
		
			rvals = np.array(rvals)		
			continf['rvals'].append(np.mean(rvals))	
		
		return continf

	#################
	elif label != None:		
		lns = []
		lbls = []		
		lines = [];	lines.extend(cntour.collections); lns = [lines[0]]
		lbls = [label]
		
		return lns,lbls

def get_clevels(dims=1):
	'''
	get clevels
	
	'''

	lvls = []
	
	for i in range(1,6):
		lvls.append(scipy.stats.chi2.cdf(i**2.,dims))
		
	lvls = np.array(lvls)
	
	
	
	return lvls[::-1]
	
def view_hist(fl,xcol,ycol):
	import sutil_sanj
	cmap = cc.cm['linear_kryw_0_100_c71_r']
	sutil_sanj.hist2d(fl[xcol],fl[ycol],bins=100,normed=True,dnorm=2,norm=LogNorm(vmin=1e-2,vmax=1),cmap=cmap,smooth=False)	

	return 

def pltfig_hist(x,y,bins=100,xrng=[0.,100],yrng=[0.,100],labels=['',''],eplot=False):
	
	'''
    NAME: pltfig_hist

    PURPOSE: scatter-plot with histograms
	

    INPUT: 
    
	x 
	y 
	bins = hist bins (default=100)
	xrng = [min,max]  (default=[0,100]) 
	yrng = [min,max]  (default=[0,100]) 
	labels = [xlabel,ylabel]  (default=['','']) 
	
    OUTPUT: plots figure
    
    REMARKS:
    


    HISTORY:   
    
    28 October 2020 (RUG)
	
	'''	

	# Set up the axes with gridspec
	fig = plt.figure(figsize=(6, 6))
	grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
	main_ax = fig.add_subplot(grid[-3:, 0:3])
	y_hist = fig.add_subplot(grid[-3:, 3], xticklabels=[], sharey=main_ax)
	x_hist = fig.add_subplot(grid[0,0:3 ], yticklabels=[], sharex=main_ax)
	
	# scatter points on the main axes
	main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
	
	main_ax.set_xlabel(labels[0])	
	main_ax.set_ylabel(labels[1])	
	main_ax.set_xlim([xrng[0],xrng[1]])
	main_ax.set_ylim([yrng[0],yrng[1]])
	
	
	def get_rp_from_e(e,ra):
		
		rp = ra/((2./(1-e)) - 1.)
		
		
		return rp		
	
	if eplot:
		e_vals = np.array([0.,0.5,0.9])
		ra = np.linspace(xrng[0],xrng[1],100)
		for e_val in e_vals:
			rp = get_rp_from_e(e_val,ra)
			main_ax.plot(ra,rp,'r--',linewidth=0.4)
			
		
	
	# histogram on the attached axes
	x_hist.hist(x, bins=bins, histtype='stepfilled',orientation='vertical', color='gray',range=[xrng[0],xrng[1]])
	# x_hist.hist(x, bins=bins, histtype='step',orientation='vertical', color='gray',range=[xrng[0],xrng[1]])


	x_hist.tick_params(labelbottom=False,labeltop=True)

	y_hist.hist(y, bins=bins, histtype='stepfilled',
	            orientation='horizontal', color='gray',range=[yrng[0],yrng[1]])

	y_hist.tick_params(labelleft=False, labelright=True)	


	return 


def addgpos(addsun=True,addgc=True,frame='hc',coord_='cartesian',markersize=5):
	
	'''
	function to overplot the locations of the Sun, Galactic center etc
	
	
	INPUT:
	addsun=True
	addgc=True
	frame='hc' ('gc')
	coord = 'cartesian' (polar)
	
	
	
	'''
	
	xsun = get_lsr()['xsun']
	Rsun = get_lsr()['Rsun']
	
	if addsun:

		if coord_ == 'cartesian':			

			if frame =='hc':
				plt.plot(xsun,0.,marker='*',markersize=markersize,color='black')	


		if coord_ == 'polar':			

			if frame =='gc':
				plt.plot(Rsun,180.,marker='*',markersize=markersize,color='black')	

	
	if addgc:	


		if coord_ == 'cartesian':			

			if frame =='gc':		
				plt.plot(0.,0.,marker='+',markersize=markersize,color='black')

		if coord_ == 'polar':			
			if frame =='gc':
				plt.plot(0,0.,marker='+',markersize=markersize,color='black')	

	
	
	return


def smoother_(image_,sigma=1,truncate=5,donoth=False):

	'''
	smooth out an image
	# sigma = standard deviation for Gaussian kernel
	# truncate = truncate filter at this many sigmas	
	
	March 15, 2022 RUG
		
	'''

	if donoth:
		return image_

	else:	
		import numpy as np
		import scipy as sp
		import scipy.ndimage	
		
	
		U = image_.copy()
		
		V=U.copy()
		V[np.isnan(U)]=0
		VV=sp.ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)
		# VV=sp.ndimage.median_filter(V,size=sigma)
		
		W=0*U.copy()+1
		W[np.isnan(U)]=0
		WW=sp.ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)
		# WW=sp.ndimage.median_filter(W,size=sigma)
				
		img=VV/WW		
		
		# mask out smear beyond data
		indf = np.where(np.isnan(image_.flatten()))[0]
		i3 = img.copy().flatten()
		i3[indf] = np.nan
		i4 = np.reshape(i3,img.shape)
			
		
		return i4

# under construction #
def smoothroutine2(i,addons):		
	
	x = addons['x']
	y = addons['y']
	kernel_epan = addons['kernel_epan']
	Deltapix_x = addons['Deltapix_x'] 
	Deltapix_y = addons['Deltapix_y'] 
	Deltapix_x_large = addons['Deltapix_x_large'] 
	Deltapix_y_large = addons['Deltapix_y_large'] 
	coords = addons['coords']

	prod_bandwidth = addons['prod_bandwidth']
	prod_bandwidth_large = addons['prod_bandwidth_large']

	this_x= coords[i][0]
	this_y= coords[i][1]
	
	#
	ker_x,n_data = kernel_epan(this_x,x,Deltapix_x)
	ker_y,n_data = kernel_epan(this_y,y,Deltapix_y)
	
	#
	ker_xy=np.sum(ker_x*ker_y)
	dens=ker_xy/(n_data*prod_bandwidth)

	#
	#----------------------------------------------------
	#
	ker_x_large,n_data=kernel_epan(this_x,x,Deltapix_x_large)
	ker_y_large,n_data=kernel_epan(this_y,y,Deltapix_y_large)



	#
	ker_xy_large=np.sum(ker_x_large*ker_y_large)
	mean_dens=ker_xy_large/(n_data*prod_bandwidth_large)
	

	over_dens_grid_local=(dens-mean_dens)/np.double(mean_dens)  
	#
	res = {}
	res['i'] = np.array([i])
	res['this_x'] = np.array([this_x])
	res['this_y'] = np.array([this_y])
	res['over_dens_grid_local'] = np.array([over_dens_grid_local])
	# # dtools.picklewrite(res,'rub1',desktop)
	
	
	return res
class smoother2_(object):

	'''

	smoothing code (Eloisa + )
	
	'''	
	def __init__(self,data=None,ncpu=5):
		print('')
		
		self.ncpu = ncpu
		self.data = data
		
	def kernel_epan(self,x_point,x_data,h):
	
	    ker=np.zeros(len(x_data)) #final kernel
	    var=(x_point-x_data)/np.double(h) #my variable
	    n_data=len(x_data); self.n_data = n_data
	    i_in=np.where(abs(var) < 1)
	    
	    if ( len(ker[i_in]) > 0):
	        ker[i_in]=(1. - (var[i_in])**2)*3./4. #(1.- (var[i_in]^2) )*3./4.
	
	    
	    return ker,n_data
		

		
	def mainstuff(self,x=[],xmin=-1,xmax=1,binsx=10,y=[],ymin=-1,ymax=1,binsy=10,meth=1):	

		xstep = (xmax - xmin)/binsx
		ystep = (ymax - ymin)/binsy

		kernel_epan = self.kernel_epan 

		xvalues=np.arange(start=xmin,stop=xmax,step=xstep) 
		yvalues=np.arange(start=ymin,stop=ymax,step=ystep) 
		
	
		nx=len(xvalues)
		ny=len(yvalues)		
		
		self.xvalues = xvalues
		self.yvalues = yvalues		
		
		xgrid=np.zeros([nx,ny])
		ygrid=np.zeros([nx,ny])
		dens_grid=np.zeros([nx,ny])
		over_dens_grid=np.zeros([nx,ny])		
		
		
		Deltapix_x=0.02 #define one value for the "local" bandwith in Z. 
		Deltapix_y=1.2 #define one value for the "local" bandwith in Vz. 
		prod_bandwidth=Deltapix_x*Deltapix_y
		
		Deltapix_x_large=4.*Deltapix_x
		Deltapix_y_large=4.*Deltapix_y
		
		prod_bandwidth_large=Deltapix_x_large*Deltapix_y_large		


		addons = {}
		addons['x'] = x
		addons['y'] = y
		addons['kernel_epan'] = kernel_epan
		addons['Deltapix_x'] = Deltapix_x
		addons['Deltapix_y'] = Deltapix_y
		addons['Deltapix_x_large'] = Deltapix_x_large
		addons['Deltapix_y_large'] = Deltapix_y_large
		addons['prod_bandwidth'] = prod_bandwidth
		addons['prod_bandwidth_large'] = prod_bandwidth_large


		iniz=-0.4 ;	fin=0.4 ; N_levels=40 ;	self.levels=np.linspace(iniz,fin,N_levels)
		if meth == 3:
			
			print('using pool..')
			ncpu = self.ncpu	
			print('running on ..'+str(ncpu))
			
			X,Y = np.meshgrid(xvalues,yvalues)
			xygrid = np.array([X.ravel(), Y.ravel()]).T
			addons['coords'] = xygrid

			indres = np.arange(xygrid.shape[0])
			# # indres = np.arange(100)
			from multiprocessing import Pool
			p=Pool(ncpu)
			overdens_smooth_intmd = p.map(partial(smoothroutine2,addons=addons), indres)
			data_pd={}
			for key in overdens_smooth_intmd[0].keys():			
				data_pd[key]=np.concatenate([d[key] for d in overdens_smooth_intmd]).transpose()	
			
			self.over_dens_grid = data_pd['over_dens_grid_local'].reshape(binsx,binsy)
			# self.xvalues = data_pd['this_x']
			# self.yvalues = data_pd['this_y']
		
			p.close()	
	
		elif meth == 4:

			for ix in np.arange(0,nx):
			    print(ix)
			    for iy in np.arange(0,ny):
			        this_x=xvalues[ix]
			        this_y=yvalues[iy]
			        #
			        ker_x,n_data = kernel_epan(this_x,x,Deltapix_x)
			        ker_y,n_data = kernel_epan(this_y,y,Deltapix_y)
			        
			        
			        
			        #
			        ker_xy=np.sum(ker_x*ker_y)
			        dens=ker_xy/(n_data*prod_bandwidth)
			        dens_grid[ix,iy]=dens
			        #
			        #----------------------------------------------------
			        #
			        ker_x_large,n_data1=kernel_epan(this_x,x,Deltapix_x_large)
			        ker_y_large,n_data1=kernel_epan(this_y,y,Deltapix_y_large)
			        #
			        ker_xy_large=np.sum(ker_x_large*ker_y_large)
			        mean_dens=ker_xy_large/(n_data*prod_bandwidth_large)
			        over_dens_grid[ix,iy]=(dens-mean_dens)/np.double(mean_dens)  
			        #
			        xgrid[ix,iy]=this_x
			        ygrid[ix,iy]=this_y 
			        self.over_dens_grid = over_dens_grid
	        

			

		def plotit(self):
			
			iniz=-0.2
			fin=0.2
			N_levels=40
			levels=np.linspace(iniz,fin,N_levels)
			cset1 = plt.contourf(smt.xvalues, smt.yvalues,smt.over_dens_grid.T, levels=levels, cmap='seismic')
			
			cbar=plt.colorbar(mappable=cset1,orientation='vertical')
			
			cbar.set_label('Overdensity', fontsize=18)
			
			# cbar.ax.tick_params(labelsize=18) 
			
			# plt.xlabel('Z (kpc)', fontsize=18)
			# plt.ylabel('Vz (km/s)', fontsize=18)
			# plt.xticks(fontsize=18)
			# plt.yticks(fontsize=18)
			# #plt.xlim([-4.5,4.])
			# #plt.ylim([-5.4,5.])
			# plt.show()
				
			plt.savefig(desktop+'/rubfigs/test.png')
			
			
			return

def usharp(myimg,radius=2,amount=200):
	
	from skimage.filters import unsharp_mask
	from skimage.filters import gaussian
	from skimage import io, img_as_float


	unsharped_img = unsharp_mask(myimg,radius=radius,amount=amount,preserve_range=True)
	
	
	
	return unsharped_img
	
#.................	
	

# spectroscopy etc
def Z2XY_abun(Z):	
	'''
	using relations given in Onno Pols et al. 1998, MNRAS 298, 525-536 
	'''
	X = 0.76 - 3*Z
	Y = 0.24 + 2*Z 	
	return X,Y
def feh2Z(feh,Zsolar=0.019,A=1.0):
	'''
	using relations given in Onno Pols et al. 1998, MNRAS 298, 525-536 
	X = 0.76 - 3*Z
	Y = 0.24 + 2*Z 
	Z = 0.76f/(1+ 3f)
	where f = 10^(M/H) *(Zsolar/Xsolar)
	'''
	Xsolar, Ysolar = Z2XY_abun(Zsolar)
	
	meh = A*feh
	f = (10**meh)*(Zsolar/Xsolar)	
	Z = (0.76*f)/(1 + 3*f)	
	
	return Z
def Z2feh(Z,Zsolar=0.019,A=1.0):
	Xsolar, Ysolar = Z2XY_abun(Zsolar)
	
	f = Z/(0.76 - 3*Z)
	meh = np.log10(f/(Zsolar/Xsolar))
	
	feh = A*meh
	
	return feh
def get_galaxia_mtip_new(log_age=1.,feh=1.): 
	
	''' 
	INPUT: (as arrays of size>1)
	log_age = log10(Age/yr)
	feh = [Fe/H]

	OUTPUT:	
	mtip_new
	'''
	import sutil_sanj
	
	HOME = getdirec('home')	
	loc = HOME+'/GalaxiaData'
	
	d = ebf.read(loc+'/feh_age_mtip.ebf','/')
	sgrid=sutil_sanj.InterpGrid(tab.npstruct(d),['feh','log_age'])
	mtip_new = sgrid.get_values('mtip',[feh,log_age],fill_value='nearest',method='linear')			
	return mtip_new

def m_h_from_feh(feh=0.,a_feh=0.):
	
	'''
	input:
	feh: [Fe/H]
	a_feh: [alpha/Fe]
	
	Output:
	[M/H]
	
	reference: Salaris & Cassisi, Evolution of Stars and Stellar Populations
	'''
	f_alpha = 10.**(a_feh)
	m_h = feh +  np.log10(0.694*f_alpha + 0.306)		
	return m_h
def feh_from_m_h(m_h=0.,a_feh=0.):
	
	'''
	input:
	m_h: [M/H]
	a_feh: [alpha/Fe]
	
	Output:
	[Fe/H]
	
	reference: Salaris & Cassisi, Evolution of Stars and Stellar Populations
	'''
	f_alpha = 10.**(a_feh)

	feh = m_h - np.log10(0.694*f_alpha + 0.306)		
	return feh
def rc_teff2clr(teff,a,b,c):	
	''' Here a,b and c are quadratic coeffecients:
	a(x^2) + bx + c '''
	cprime = (c) - (5040./teff)
	discrim_sqrt = np.sqrt((b**2.0) - (4*a*cprime))
	clr = (-b + discrim_sqrt)/(2*a)		
	
	return clr	
def rc_clrzero(feh,teff,method='direct'):
	''' RETURNS jk_0 FROM [Fe/H] & Teff	'''
	
	HOME = getdirec('phddir')
	glxdir = getdirec('galaxia')
		
	if method == 'direct':
				
		print('')
	

		############################################
		## [teff, Feh] to clr (direct method)
		############################################	
		import calibrations_direct		

		calib_file = 'k18rc_jk_teff_feh_coeff.pkl'
		

		# popt_pop = ebf.read(HOME+'/GalaxiaWork/calibrations/now_direct/calib_res_direct.ebf')['clumps']
		popt_pop = dtools.pickleread(glxdir+'/'+calib_file)
		popt = np.array([popt_pop[1],popt_pop[2],popt_pop[3],popt_pop[4],popt_pop[5],popt_pop[6]])
		popt = popt.astype(np.float)
	
		xs = np.array([feh,teff])
		clr = (calibrations_direct.func(xs,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],prnt=True))	
				
		
	elif method == 'inverted':
		
		############################################
		## teff2clr (inverted method)
		############################################					
		
		pop = 'clumps'; usemet='n'
		out=np.loadtxt(HOME+'/GalaxiaWork/calibrations/calib_res.txt', dtype=str,usecols=(0,1,2,3,4,5,6,7,8))	
		for i in range(len(out)):
			if pop in out[i,0] and usemet in out[i,1]:			
				c,b,a = float(out[i,2]),float(out[i,3]),float(out[i,4])			
		
		clr = rc_teff2clr(teff,a=a,b=b,c=c)	
	
	return clr
def distance_rc(j,ks,teff,logg,feh,akval=[]):
	'''
	1. f(i) = A(i)/E(B-V); use GALAXIA tables --> UKIRT J & K        
	2. red-clump Mk-[Fe/H] curve valid for -0.8<[Fe/H]<0.4
	
	''' 
	print('using new Ak correction')
	print('')
	print('reading Mks_[Fe/H] interp file')
	glxdir = getdirec('glxdir')
	loc = glxdir+'/calibrations'
		
	nam_intp_file = 'absks_feh_intp_tmasswise_direct'			
	absks_feh_intp = ebf.read(loc+'/'+nam_intp_file+'.ebf')
	f = interp1d(absks_feh_intp['feh_intp'],absks_feh_intp['absks_intp'],bounds_error=False,fill_value=(absks_feh_intp['absks_intp'][0],absks_feh_intp['absks_intp'][-1]))			
	#mk_intp = np.interp(feh,absks_feh_intp['feh_intp'],absks_feh_intp['absks_intp'])	- correct this - requires sorted arrays!!
	mk_intp = f(feh)
	#print('mk_curve is :')
	#print(mk_intp)
	
	#print(absks_feh_intp)	
	#plt.plot(feh,mk_intp,'r.')


	aebv = aebv_galaxia() 

	fj=aebv.info('TMASSWISE_J')[2]  #0.88579 
	fk=aebv.info('TMASSWISE_Ks')[2]  #0.36247	
	
	if akval == []:
		clr = rc_clrzero(feh,teff,method='direct')	
		ak=fk*((j-ks)-clr)/(fj-fk)		
	else:
		ak = akval[0]

	
	return  autil.dmod2dist(ks-mk_intp-ak)
def selfunc_rc(teff,logg,feh,mthd='direct',clr=[]):		
	print('selecting red-clumps (Bovy"s method but using abs_j,ks etc)')	
	
	zsolar = 0.019
	mydata = {'teff':teff,'logg':logg,'feh':feh}
	mydata['Zmet'] = (10**(mydata['feh']))*zsolar		
	
	if clr == []:
		mydata['clr'] = rc_clrzero(mydata['feh'],mydata['teff'],method=mthd)

	else:
		print('using true colours')
		mydata['clr'] = clr[0].copy()

		
	############################################
	## SELECT RED-CLUMPS
	############################################		
	teff_ref = -(382.5*mydata['feh']) + 4607
	loggup = 0.0018*(mydata['teff'] - teff_ref) + 2.5	
	zlower = 1.21*((mydata['clr'] -0.05)**9.0) + 0.0011;
	zupper = 2.58*((mydata['clr'] -0.40)**3.0) + 0.0034								
	
	
	cond1 = (mydata['clr'] > 0.5)&(mydata['clr'] < 0.8) 
	cond2 = (mydata['logg']>= 1.8)&(mydata['logg']<= loggup)
	cond3 = (mydata['feh']>=-0.8)&(mydata['feh']<= 0.4)
	cond4 = (mydata['Zmet']<= 0.06)&(mydata['Zmet']>=zlower)&(mydata['Zmet']<=zupper) 
	#cond4 = 1	

	cond = cond1&cond2&cond3&cond4        
	return cond


# Fourier
def psd2d_old(image,binsize,dx,dy,pad=0):
	import bovy_psd
	
	'''
	OLD VERSION
	'''	
	
	#image=image-np.mean(image[True-np.isnan(image)])
	image=image-np.mean(image[True ^ np.isnan(image)])	
	image[np.isnan(image)]= 0.
	
	image=image-np.mean(image)
	temp= np.fft.fftshift(np.fft.fft2(image,s=((1+pad)*image.shape[0]+1,(1+pad)*image.shape[1]+1)))
	if pad == 0:
		temp= np.fft.fftshift(np.fft.fft2(image,s=((1+pad)*image.shape[0],(1+pad)*image.shape[1])))
	
	nxbins = image.shape[0]
	nybins = image.shape[1]
	
	Lx = image.shape[0]*dx
	Ly = image.shape[1]*dy
	Lbovx = 6.75
	Lbovy = 6.75

	#nom_cor = dx*dy*0.01716   #64./(((nxbins*nybins)**2.)*dx*dy)                   # to match Bovy 2015
	#nom_cor =4./((nxbins*nybins)**2.)                           
	#bv_cor = 16. * (((Lx*Ly)/(Lbovx*Lbovy))**2.)
	#hf=np.power(np.abs(temp),2.0)*nom_cor*bv_cor
	
	   	                    	
	hf=np.power(np.abs(temp),2.0)*(dx*dy)*0.01716  	                    
	
	
	x=np.fft.fftshift(np.fft.fftfreq(temp.shape[0],dx))
	y=np.fft.fftshift(np.fft.fftfreq(temp.shape[1],dy))
	
	binsize=binsize/(temp.shape[0]*dx)   
	
	x2,y2=np.meshgrid(x,y,indexing='ij')
	r = np.hypot(x2, y2)
	nbins = int(np.round(r.max() / binsize)+1)
	maxbin = nbins * binsize
	
	h1=sutil.hist_nd(r.flatten(),bins=nbins,range=[0,maxbin])
	res=h1.apply(hf.flatten(),np.mean)
	loc=h1.locations[0]
	
	return loc*2.1,res
def psd2d(image,binsize,dx,dy,pad=0):
	import bovy_psd
	'''
	FROM SANJIB: OCTOBER 01, 2018
	'''
	

	fac=np.sum(np.isfinite(image))*1.0/np.prod(image.shape); #print('fac = '+str(fac)); print('imshape = ='+str(np.prod(image.shape)))
	image=image-np.nanmean(image)
	image[np.isnan(image)]= 0.0
	temp= np.fft.fftshift(np.fft.fft2(image,s=((1+pad)*image.shape[0]+1,(1+pad)*image.shape[1]+1)))
	hf=(np.power(np.abs(temp),2.0)*dx*dy/(np.prod(image.shape)*fac))
	x=np.fft.fftshift(np.fft.fftfreq(temp.shape[0],dx))
	y=np.fft.fftshift(np.fft.fftfreq(temp.shape[1],dy))    
	binsize=binsize/(temp.shape[0]*dx)
	x2,y2=np.meshgrid(x,y,indexing='ij')
	r = np.hypot(x2, y2)
	nbins = int(np.round(r.max() / binsize)+1)
	# # print('testing................')
	# # print(nbins)
	# # print('.........')
	maxbin = nbins * binsize
	h1=sutil.hist_nd(r.flatten(),bins=nbins,range=[0,maxbin])
	res=h1.apply(hf.flatten(),np.mean)
	loc=h1.locations[0]
	
	#print np.sum(hf)*(x[1]-x[0])*(y[1]-y[0]),np.sum(image*image)/(np.prod(image.shape)*fac),np.trapz(res[0:-1]*loc[0:-1]*2*np.pi,loc[0:-1]),fac
	#ind = np.where(np.isfinite(res))[0]
	#print(np.sum(res[ind]*2.*np.pi*loc[ind]*(loc[1]-loc[0])) )
	return loc*2.1,res



# vizier
def get_vizier(catalog):
	nam = {'gaiadr2':'vizier:I/345/gaia2',
		'gaiaedr3':'vizier:I/350',	
		'wise':'vizier:II/311/wise',	
		'allwise':'vizier:II/328/allwise',	
		'unwise':'vizier:II/363/unwise',	
		'catwise':'vizier:II/365/catwise',	
		'panstarrs':'vizier:II/349/ps1',	
		'sdss':'vizier:V/147/sdss12',
		'harris': 'vizier:VII/202/catalog',
		'spitzer':'vizier:II/293/glimpse',	
		'dodddr3':'J/A+A/670/L2',
		'twomass':'vizier:II/246/out'}	
	
	return nam[catalog]
def mk_upload(data,FILENAME='tst',FORMAT='csv',loc='',rakey='ra',deckey='dec',**kwargs):		
	'''
	creates .csv file 
	
	'''	
	from astropy.table import Table, Column
	from astropy.io import ascii	


	mydata={}
	mydata[rakey] = data[rakey]
	mydata[deckey] = data[deckey]
	idkey = kwargs['idkey']
	mydata[idkey] = data[idkey]
	
	#mydata['_2mass'] = data['_2mass']
	
	tab = Table(mydata) 
	
	if loc == '':
		loc = os.getcwd()

	fullname =loc+'/'+FILENAME+'.'+FORMAT 	 
	tab.write(fullname,format='csv',names=list(mydata.keys()),overwrite=True)	
	
	print('')
	print('file written')
	return tab	
def cmatch(data,radius=5.,cat2='vizier:I/345/gaia2',colra1='ra',coldec1='dec',outname='mydata',match_key='source_id',loc='',write_as='fits'):
	from astropy import units as u
	from astroquery.xmatch import XMatch
	from astropy.table import Table, Column
	from astropy.io import ascii
	
	if loc == '':
		loc = os.getcwd()

	print('')
	print('Cross-matching with '+cat2)
	mk_upload(data,rakey=colra1,deckey=coldec1,idkey=match_key,loc=loc)	


	data_temp = XMatch.query(cat1=open(loc+'/tst.csv'),cat2=cat2,max_distance=radius * u.arcsec,colRA1=colra1,colDec1=coldec1)
	#table = XMatch.query_async(cat1=open(cat1),cat2=cat2,max_distance=radius * u.arcsec,colRA1=colra1,colDec1=coldec1)

	print(data_temp)
	kyuse = ''
	for ky in data_temp.keys():
		if ky.startswith('ra'):
			kyuse = ky

	print(data_temp[ky].size)		
	print('joining...')
	tabpy.ljoin(data,data_temp,match_key,match_key)
	
	for ky in data.keys():
		if ky.startswith('ang'):
			tabpy.sortby(data,ky)		   	

	tabpy.unique(data,match_key)

	print('')
	print('----------')	
	if write_as == 'fits':
		dt = Table(data)
		dt.write(loc+'/'+outname+'.fits',format='fits',overwrite=True)	
		print('File written to '+loc)
	if write_as == 'ebf':
		ebfwrite(data,outname,loc)
		print('File written to '+loc)		


	
	return data	
def topcatxm(in1,in2,outfile):
	
	print('')
	
	os.system('topcat -stilts tmatch2 in1='+in1+' in2='+in2+'\
	           join=1and2 matcher=exact values1=source_id values2=source_id find=best ofmt=colfits-plus out='+outfile)


	return	



def xmatch_topcat(floc1,floc2,floc3,radius=5):
	'''
	all with full path:
	floc1 = first file
	floc2 = second file
	floc3 = save file
	radius = arcseconds
	'''
	
	os.system('topcat -stilts tskymatch2 in1='+floc1+' in2='+floc2+' ra1=ra dec1=dec ra2=ra dec2=dec join=1and2 find=best error='+str(radius)+' ofmt=colfits-plus out='+floc3)

	return

# Extinction


			
def redden_fac(maptype=''):
	'''
	
	maptype: '' == uses Galaxia values
	maptype: 'bayestar3d' == uses values as in Bayestar
	
	HISTORY: 14 November 2019 (RUG)
	         21 JUNE 2021 (RUG)
	
	Purpose: Returns multiplicative factor to convert reddening E(B-V) 

	1. from Cassagrande & Vandenburg 2018, for Gaia bands
	2. from Green et al. 2019, in to extinction for PANSTARS and 2MASS bands.	
	
	'''


	aebv=aebv_galaxia()
	

	val=   {'jmag': aebv.info('TMASSWISE_J')[2],
		    'hmag': aebv.info('TMASSWISE_H')[2],
		    'kmag': aebv.info('TMASSWISE_Ks')[2],
		    'phot_g_mean_mag': 2.74,
		    'bp': 3.374,
		    'rp': 2.035}


	if maptype == 'bayestar3d':
		val['gP1'] =  3.518
		val['rP1'] = 2.617
		val['iP1'] = 1.971
		val['zP1'] = 1.549
		val['yP1'] = 1.263		
		val['jmag'] = 0.7927
		val['hmag'] = 0.4690
		val['kmag'] = 0.3026



	wise='WISE_W'
	for num in [1,2,3,4]:
		# val['w'+str(num)+'mag']=aebv.info(wise+str(num))[2]
		val['w'+str(num)+'mag']=aebv.info('TMASSWISE_W'+str(num))[2]
		val['w'+str(num)+'mpro']=aebv.info('TMASSWISE_W'+str(num))[2]
		        
	# val['w1mpro'] = val['bp']/125.6  
	# val['w2mpro'] = val['bp']/138.5  
	return val

def deredden_map():
	mags={'jmag':'2mass_j', 'kmag':'2mass_ks','hmag':'2mass_h', 'w1mag':'wise_w1', 'w2mag':'wise_w2', 'w3mag':'wise_w3', 'w4mag':'wise_w4', 'phot_g_mean_mag':'gaia_g'}
	
	return mags
class aebv_galaxia(object):

	'''
    NAME:

    PURPOSE: aebv factors from GALAXIA


    INPUT: bandname

    OUTPUT: 'name, lambda, fac'
       
    HISTORY: 10 April 2020

	'''	
	def __init__(self):
		

		filename=getdirec('galaxia')+'/Isochrones/aebv_factor.ebf'
		self.aebv_file = ebf.read(filename)
		self.aebv_file['filter'] = self.aebv_file['filter'].astype(str)
		
		self.get_bands()
	def get_bands(self):
		
		val = self.aebv_file['filter']
		return val
					
	def info(self,band):
		
		if band in self.aebv_file['filter']:
			
			indf = np.where(self.aebv_file['filter']==band)[0]
			#print('name, lambda, fac')
			return self.aebv_file['filter'][indf][0], self.aebv_file['lambda_eff'][indf][0],self.aebv_file['aebv_factor'][indf][0]
			
		else:			
			raise Exception('check bandname')
def deredden(glon,glat,mag1=None,bandname1=None,mag2=None,bandname2=None,getebv_only=False,corrfac='bstar'):	
	
	'''
	Estimate extinction at infinity using Schlegel maps
	
	
	corrfac = bstar (0.86 by default)
			  '' (uses the scheme in Sharma 2011, Koppelman 2020)	
	'''
	
	import gutil2	
	filename=getdirec('galaxia')+'/Isochrones/aebv_factor.ebf'
	


	ext=gutil2.Extinction(GalaxiaPath=getdirec('galaxia')+'/Extinction/',filename=filename)
	ebv=ext.ebv_inf(glon,glat)

	if corrfac == 'bstar':	
		ebv = ebv*0.86
	elif corrfac == 'koppelman':
				
		indcor = np.where(ebv > 0.15)[0]
		
		'''
		correct for overestimated Schlegel maps using the scheme in Sharma 2011 etc
		
		'''
		cor_fac = 0.6 + (0.2*(1 - np.tanh((ebv[indcor] - 0.15)/0.3) ) )		
		ebv[indcor] = ebv[indcor]*cor_fac


	if getebv_only:
		return ebv		

	
	else:
	
		aebv_file = ebf.read(filename)
		filter_names = (np.array([filters.lower() for filters in aebv_file['filter']])).astype(str)
		
		bandnames = {'bandname1':bandname1}
		if mag2 is not None:
			bandnames['bandname2'] = bandname2
	
		for bandname in bandnames.keys():
			if bandnames[bandname] not in filter_names:
				print(bandnames[bandname])
				print(aebv_file['filter'])
				bandname = raw_input('bandname = ')	;	# bandname = input('bandname = ')		
				bandnames[bandname] = bandname
	
		print('')
		print('dereddenning:')
		print(bandnames)
			
	
		col0=ext.deredden(ebv,mag1,bandname1)
		
		if mag2 is not None:
			col0=ext.deredden(ebv,mag1,bandname1)-ext.deredden(ebv,mag2,bandname2)				
		
		return col0, ebv

def extmapsanj(l,b,r):
	
	'''
	function to interpolate extinction from 2D and 3D maps
			
    NAME: extmapsanj

    PURPOSE: obtain ebv using Sanjib's interpolated extinction map

    INPUT: l,b,r

    OUTPUT: ebv-2d, ebv-3d, intfac
       
    HISTORY: August 16, 2022
	
	'''	
	l = np.array(l)
	b = np.array(b)
	r = np.array(r)

	data = {}
	data['glon'] = l.copy()
	data['glat'] = b.copy()
	data['rad'] = r.copy()
		
	import scipy.interpolate

	GalaxiaPath=getdirec('galaxia')+'/Extinction/'
	x=np.zeros((data['glon'].size,3),dtype='float64')
	x[:,0]=data['glon']
	x[:,1]=data['glat']
	x[:,2]=np.log10(data['rad'])
	data3d=ebf.read(GalaxiaPath+'ExMap3d_1024.ebf','/ExMap3d.data')
	xmms3d=ebf.read(GalaxiaPath+'ExMap3d_1024.ebf','/ExMap3d.xmms')
	points3d=[xmms3d[i,0]+np.arange(data3d.shape[i])*xmms3d[i,2] for i in range(data3d.ndim)]
	data2d=ebf.read(GalaxiaPath+'Schlegel_4096.ebf','/ExMap2d.data')
	xmms2d=ebf.read(GalaxiaPath+'Schlegel_4096.ebf','/ExMap2d.xmms')
	points2d=[xmms2d[i,0]+np.arange(data2d.shape[i])*xmms2d[i,2] for i in range(data2d.ndim)]
	
	temp3d=scipy.interpolate.interpn(points3d,data3d,x,bounds_error=False,fill_value=None,method='linear')
	temp2d=scipy.interpolate.interpn(points2d,data2d,x[:,0:2],bounds_error=False,fill_value=None,method='linear')
	data['exbv_schlegel_inf']=temp2d
	data['exbv_schlegel']=temp2d*temp3d	
	
	return 	temp2d, temp2d*temp3d, temp3d	



class deredden_3d(object):

	'''
	
    NAME: deredden_3d from schlegel 

    PURPOSE: estimate 3d extinction from 2d Schlegel maps by correcting using the methods in Binney 2014 + Koppelman 2020 and Sharma 2011

	
	** has to be done star-by-star!!

    INPUT:
			l (degrees)
			b (degrees)
			d (kpc)
			mag
			mag_map

    OUTPUT: a_val, mag0, intfac(ratio)
       
    HISTORY: 28 April 2020 [Groningen]
           : 02 September 2022 [Torino]
	
	need to add a line to apply only to R > Rwarp and Rflare

	'''	
	def __init__(self):
		
		
		self.constants()		
			
	
	def constants(self):
		
		
		self.Rsun = get_lsr()['Rsun']
		self.zsun = get_lsr()['zsun']
		self.hr = 4.2 #kpc
		self.hz = 0.088 #kpc
		self.yflare = 0.0054 #kpc^-1
		self.ywarp = 0.18 #kpc^-1
		
		self.Rflare = 1.12*self.Rsun #kpc
		self.Rwarp = 8.4 #kpc
		
		
		return
		
		
	def dum(self,rterm):
		l = self.l
		b = self.b
		
		R,lgc = self.getcoords(l,b,rterm)		
		
		self.kflare = 1. + (self.yflare*(np.min([self.Rflare,R-self.Rflare])  ))
		self.zwarp= (self.ywarp*(np.min([self.Rwarp,R-self.Rwarp])  )*np.sin(np.radians(lgc)))
		
		a1 = (np.cos(np.radians(b)))**2.
		a2 = (-2.)*self.Rsun*(np.cos(np.radians(b)))*(np.cos(np.radians(l)))
		a3 = (self.Rsun)**2.
		a4 = (np.sin(np.radians(b)))
		
		tmp1 = np.exp(self.Rsun/self.hr)		
		tmp2 = np.exp(-(( (a1*rterm**2.) + (a2*rterm) + a3)**0.5)/self.hr)
	
	
		tmp3 = np.exp(-(abs(  (a4*rterm) + self.zsun - self.zwarp)  )/(self.kflare*self.hz) )
		
		val = tmp1*tmp2*tmp3
		
	
		return val
			
	
	def getcoords(self,l,b,distance):
		
		
		xsun = get_lsr()['xsun']
		zsun = get_lsr()['xsun']
		
		x,y,z = autil.lbr2xyz(l,b,distance)
		xgc = x+xsun
		zgc = z+zsun
		ygc = y.copy()
		
		#rgcd = np.sqrt(xgc**2. + ygc**2. + zgc**2.)
		rgc = np.sqrt(xgc**2. + ygc**2.)
		
		lgc = np.degrees(np.arctan2(ygc,xgc))		
		
		return rgc,lgc
		
	def getintfac(self,l,b,r):
		
		# timeit = stopwatch()
		# timeit.begin()
		
		self.l = l
		self.b = b
		self.r = r
		
		func = lambda rfn : self.dum(rfn)
		ts = scipy.integrate.nquad(func,ranges=[[0.,r]],full_output=True)

		# print('here...')
		# timeit.end()

		
		ts1 = scipy.integrate.nquad(func,ranges=[[0.,np.inf]],full_output=True)

		# print('here...')
		# timeit.end()

					
		self.intfac = ts[0]/ts1[0]
		return self.intfac
		
	def getmag0(self,mag,mag_map):
		'''
		returns: mag0_3d, mag0_2d, Amag_3d
		'''
		
		l,b,r = self.l,self.b,self.r
				
		## calculate a_lambda
		mag0_sch,ebv_sch = deredden(l,b,mag,mag_map)
						
		a_mag_sch = mag - mag0_sch
		
		a_mag_sch_3d = self.intfac*a_mag_sch
		mag0_sch_3d = mag - a_mag_sch_3d
		
		# return a_mag_sch_3d, mag0_sch_3d, intfac, ebv_sch
		return mag0_sch_3d, mag0_sch, a_mag_sch_3d 

	
def chk():	
	l,b,r = 187.36485163807515, -35.11673792906733, 0.6573329467773438	
	myred = deredden_3d()  
	myred.getintfac(l,b,r)
	
	return 
		

def dustmap_web(l,b,d,band,version='bayestar2019'):
	'''
	estimate extinction from 3d maps from Green et al. 2019 using their webserver
	
	l (degrees)
	b (degrees)
	d (kpc)

	'''
	from astropy.coordinates import SkyCoord
	import astropy.units as u
	from dustmaps.bayestar import BayestarWebQuery 
	
	l = l*u.deg
	b = b*u.deg
	d = (d*1000.)*u.pc
	
	coords = SkyCoord(l, b, distance=d, frame='galactic')
	
	q = BayestarWebQuery(version=version)
	
	E = q(coords, mode='median')
	print('E is ...'+str(E))
	
	rfac = redden_fac()
	if band in rfac.keys():		
		A_lambda = E*rfac[band]
	else:
		A_lambda = np.zeros(len(l)) + np.nan
	
	return A_lambda

class dustmap_green(object):

	'''
	estimate extinction from 3d maps from Green et al. 2019 using their downloaded map
	l (degrees)
	b (degrees)
	d (kpc)
	
	machine: home or huygens
	'''	
	def __init__(self,machine='home',usemap='bayestar'):
		print('')
		
		self.usemap = usemap
		self.machine = machine
		print('running on '+self.machine)
		
		if machine == 'kapteyn':
			self.readloc_ = '/net/huygens/data/users/khanna/Documents/phd_work'
		if machine == 'bologna':
			self.readloc_ ='/home/guest/Sunny/data8/Documents/pdoc_work'
		if machine == 'cluster2':
			self.readloc_ =getdirec('pdocdir',at='cluster2')
					
	def loadit(self,max_samples=20):
						
		from dustmaps.bayestar import BayestarQuery
		timeit=stopwatch()
		timeit.begin()
			
		maploc = 'dustmaps/'+self.usemap
		mapname = os.listdir(self.readloc_+'/'+maploc)[0]
		
		self.bayestar = BayestarQuery(max_samples=max_samples,map_fname=self.readloc_+'/'+maploc+'/'+mapname)


		timeit.end()	
		return 
		
	def get(self,l,b,d,band=None,mode='median',getebv_only=False,getebv=False):
		
		from astropy.coordinates import SkyCoord
		import astropy.units as u		
	
		l = l*u.deg
		b = b*u.deg
		d = (d*1000.)*u.pc
		
		coords = SkyCoord(l, b, distance=d, frame='galactic')
		
		q = self.bayestar 
		
		if mode == 'median':
			E = q(coords, mode=mode)
		elif mode == 'samples':
			E = q(coords, mode='samples')	
			
		if getebv_only:
			
			return E
		
		else:
			
			rfac = redden_fac(maptype='bayestar3d')
			
			if band in rfac.keys():				
				A_lambda = E*rfac[band]
			elif l.size == 1:
				A_lambda = np.nan
			else:
				A_lambda = np.zeros(len(l)) + np.nan
	
			if getebv:
				print(' A_lambda & E(B-V) ')
				return A_lambda, E		
			else:
				return A_lambda


def extmaplallemant(l,b,r,typ='19'):
	
	
	'''
	l [deg]
	b [deg]
	r [kpc]
	
	typ = '18' or '22'
	
		
	'''
	import imp


	import lallement as llm

	
	
	if typ =='18':
		lmap = llm.L18map(dustmaploc=getdirec('pdocdir')+'/dustmaps')
	elif typ =='19':
		lmap = llm.L19map(dustmaploc=getdirec('pdocdir')+'/dustmaps')
		
	ebv = lmap.getebv(l,b,r,plot=True)	
	
	return ebv
	
	



# Other Astronomy (rarely used)
def periodic_dist(x1,x2,l=360.):
	
	'''
	To get stars around  longitude=20.0 use the following function
	periodic_dist(l,20,360) < 20	
	
	'''
	
	dx=(x2-x1)%360.0
	if (hasattr(dx,"__len__") == False):
		return np.min([dx,l-dx])
	else:
		return np.select([dx>l*0.5],[l-dx],default=dx)
def a2age(age_a):
	from astropy.cosmology import Planck15	
	z = (1.0/age_a) - 1.0
	
	age_giga = Planck15.lookback_time(z).value
	
	return age_giga
def get_absmag(mag,a_ext,dmod):
	
	'''
	Absolute magnitude from apparent magnitude and dmod
	
	'''
	
	abs_mag = mag - a_ext - dmod
	
	return abs_mag
def get_absmag_d(mag,a_ext,d):
	
	'''
	Absolute magnitude from apparent magnitude and distance 
	
	
	'''
	
	import uncertainties.unumpy as unumpy
	
	abs_mag = mag - a_ext - 10 + (5.*unumpy.log10(1./d))		

	
	return abs_mag
def get_absmag_plx(mag,a_ext,parallax):
	
	'''
	Absolute magnitude from apparent magnitude and parallaxes
	
	'''
	# import uncertainties.unumpy as unumpy
	
	# indx = np.where(parallax<0)[0]
	# parallax[indx] = np.nan
	
	# abs_mag = mag - a_ext - 10 + (5.*unumpy.log10(parallax))	
	
	abs_mag = mag - a_ext - 10 + (5.*np.log10(parallax))	
	
	# return abs_mag.astype(np.float32)
	return abs_mag
	
def get_dmod(abs_mag,a_ext,mag):
	
	'''
	dist modulus from Absolute magnitude 	
	'''
	
	dmod = mag - a_ext - abs_mag
	return dmod
	
def phi_mod(value):
	
	'''
	Modified Galactocentric phi (to match Katz 2018 GDR2)
	'''

	#val = np.degrees(np.arctan2(y,x))
	
	phi=[]
	for val in value:	
		if val >= 0.:
			val = 180. - val 			
		elif val <= 0.:
			val = -1.*(180. - abs(val))
		phi.append(val)
	
	return np.array(phi)
def iter_cent(part_data,rcentmax):
	## Iterative centering to return new x_cen ##
	r_max = max(abs(part_data[:,0]))
	ind=np.arange(part_data[:,0].size)	
	while (r_max > rcentmax):		
		xyz_cen = [np.mean(part_data[ind,0]),np.mean(part_data[ind,1]),np.mean(part_data[ind,2])]	
		r_new=0					
		for i in [0,1,2]:
			temp=(part_data[:,i]-xyz_cen[i])						
			r_new+=temp*temp	
		r_new = np.sqrt(r_new)				
		ind=np.where(r_new < r_max)[0]				
		r_max = 0.7*r_max			
	
	return xyz_cen, ind


# Healpix
def allsky_area():
	
	'''
	returns the allsky surface area in degrees squared
	'''
	
	r = 360./(2.*np.pi)
	A = 4*np.pi*(r**2.)
	
	return A
def pixsize2nside(pixel_size):
	
	'''	
	
		
    NAME: pixsize2nside

    PURPOSE: pixel size to nside


    INPUT: pixel_size [degrees squared]

    OUTPUT: nside
       
    HISTORY: March 05, 2020 (Groningen)       	
	
	
	'''
		
	import healpy as hp
	
	A_allsky = allsky_area() 
	npix = (int(A_allsky/pixel_size))
	nside = int(np.sqrt((npix/12.)))

	# nside = hp.npix2nside(int(A_allsky/pixel_size))
	
	return nside




def sourceid2hpx(source_id,level=12):
	'''
	source_id (has to be array or int)
	level (12 by default)
	
	hpx id 
	
	April 23, 2021
	February 18, 2022
	'''	
	
	# val = int(source_id/((2**35)*(4**(12-level))))
	val = (source_id/((2**35)*(4**(12-level))))

	# if val.size > 1:
		# return val.astype(int)
	# else:	
		# return int(val)
	# if type(val) != 'float'.size > 1:
		# return val.astype(int)
	# else:	
		# return int(val)

	if isinstance(val, (float)) ==  False:
		return val.astype(int)
	else:

		return int(val)
	
def hpx2sourceid(hpx,level=12):
	'''
	hpx id 
	level (12 by default)
	
	source_id id 
	
	May 06, 2021
	'''	
	
	
	source_id = hpx*((2**35)*(4**(12-level))) 
	

	# if source_id.size > 1:
		# return source_id.astype(int)

	# else:	
		# return int(source_id)
		
	return int(source_id)




def sourceid2pos(source_id,level=12):
	
	'''
	source_id 
	level (12 by default)
	
	ra, dec, l , b
	
	May 06, 2021
	'''
	
	from astropy_healpix import HEALPix
	from astropy import units as u  
	
	nside = 2**level	
	hp = HEALPix(nside, order='nested')	
	val = (source_id/((2**35)*(4**(12-level))))
	val = np.array(val)
	
	if val.size>1:
		val = val.astype(int)
	else:
		val = int(val)	

	
	ra, dec = hp.healpix_to_lonlat([val])  # radians	
	ra,dec = np.degrees(ra.value)[0], np.degrees(dec.value)[0]
	
	l,b = autil.equ2gal(ra,dec) 
		
	return ra, dec, l, b 

def hpix2pos(hpix,nside,nest=True,lonlat=True,style=''):

	'''

	
	hpixel
	nside
	
	ra, dec, l , b
	
	* note if style == 'gaia' healpix id is passed in, it returns ra dec instead of l, b, internally corercted.
	
	HISTORY: 20 May, 2024 (INAF-TORINO)
	
	'''	
	
	l,b = healpy.pix2ang(nside,hpix,lonlat=lonlat,nest=nest)	
	ra,dec = autil.gal2equ(l,b) 	

	if style == 'gaia':
		ra,dec = healpy.pix2ang(nside,hpix,lonlat=lonlat,nest=nest)	
		l,b = autil.equ2gal(ra,dec) 	
		
	return ra, dec, l, b 


def DeclRaToIndex(decl, RA, nside):
    return healpy.pixelfunc.ang2pix(nside,np.radians(decl+90.),-np.radians(360.-RA))
def hpstatistic(nside, ra, dec, statistics='count', vals=None,nest=False):
    """
    Create HEALpix map of count, frequency or mean or rms value.
    :param nside: nside of the healpy pixelization
    :param v: either (x, y, z) vector of the pixel center(s) or only x-coordinate
    :param y: y-coordinate(s) of the center
    :param z: z-coordinate(s) of the center
    :param statistics: keywords 'count', 'frequency', 'mean' or 'rms' possible
    :param vals: values (array like) for which the mean or rms is calculated
    :return: either count, frequency, mean or rms maps
    """
    
    import healpy
    npix = healpy.nside2npix(nside)
    pix = healpy.ang2pix(nside,ra, dec,lonlat=True,nest=nest)
          
    
    
    n_map = np.bincount(pix, minlength=npix)
    if statistics == 'count':
        v_map = n_map.astype('float')
    elif statistics == 'frequency':
        v_map = n_map.astype('float')
        v_map /= max(n_map)  # frequency [0,1]
    elif statistics == 'mean':
        if vals is None:
            raise ValueError
        v_map = np.bincount(pix, weights=vals, minlength=npix)
        v_map /= n_map  # mean
    elif statistics == 'rms':
        if vals is None:
            raise ValueError
        v_map = np.bincount(pix, weights=vals ** 2, minlength=npix)
        v_map = (v_map / n_map) ** .5  # rms
    else:
        raise NotImplementedError("Unknown keyword")
    return v_map, pix
	

# tepper sims

#-- DICE FILES USING FORTRAN
def tep(snp,svloc=''):
	from astropy.table import Table, Column
	from astropy.io import ascii	
	
	print('svloc = '+svloc)
	print(os.getcwd()+'/'+svloc)
	
	loc = os.getcwd()
		
	dt = ascii.read(loc+'/table.txt')
	kys = {'x[kpc]':'px', 'y[kpc]':'py', 'z[kpc]':'pz','vx[km/s]':'vx','vy[km/s]':'vy','vz[km/s]':'vz','ID':'part_id'}
	
	data = {}
	for key in kys:
		data[kys[key]] = np.array(dt[key])
	

	ebfwrite(data,snp,os.getcwd()+'/'+svloc)
	return

def runtep(typ=3):
	'''
	
	FORTRAN 
	
	'''
	
	
	timeit = stopwatch()
	timeit.begin()		
	
	desc = {3:'thick_disc',11:'thin_disc'}
	print('making '+desc[typ]+'...')	
	typ = str(typ)	
	
	# previously did for 60 steps
	#outs = np.linspace(1.,300.,300) # iso	
	#outs = np.linspace(1.,174.,174) # s
	outs = np.linspace(1.,154.,154)  # r

	print(len(outs))
	print(outs)
	outs = outs.astype(int)
	outps = []
	for t in outs:		
		val = t.astype(str).rjust(5,'0')
		outps.append(val)
	outps = np.array(outps)	

	simname = 'dice_MW_Sgr_dSph_r_hires' #dice_MW_Sgr_dSph_p_MWonly_hires
	svloc = simname+'_'+typ
	os.mkdir(svloc)
	for out in outps:		
		print(out)
		
		dloc = '/import/photon1/tepper/ramses_output/Nbody/dice/myics/'+simname
		cf = './read_part -inp '+dloc+'/output_'+out+' -out table.txt -typ '+typ+' -nc 11 -fil ascii'
		os.system(cf)
		tep(out+'_'+typ,svloc=svloc)



	timeit.end()		


#-- AGAMA FILES USING pynbody


def read_tepper_pynbody(file1,clock=True):
	'''
	pyNbody 
	

	Access to the following models can be found on:
	
	/export/photon1/tepper/ramses_output/Nbody/agama/myics/
	

	I. Models with a bar:	
	agama_test	
	agama_test_hires
	
	II. Featureless model:	
	chequers_2018_iso
	 	
	AGAMA component IDs
	1 -> DM halo	
	3 -> stellar bulge	
	5 -> thick disc	
	7 -> thin disc	
	
	
	'''
	
	import pynbody
	
	if clock:
		timeit = stopwatch()
		timeit.begin()	

		
	MaxCompNumber = 11
	ro = pynbody.load(file1)
	ro.physical_units()
	tstamp = float(ro.properties['time'])                               # in Gyr
	ro['part_id']=ro['iord'] % (2*MaxCompNumber)
	ro['pos'] = ro['pos'] - (ro['pos']).mean(axis=0)
	ro['vel'] = ro['vel'] - (ro['vel']).mean(axis=0)
	
	d={}
	d['px']=np.array(ro['pos'][:,0])
	d['py']=np.array(ro['pos'][:,1])
	d['pz']=np.array(ro['pos'][:,2])
	d['vx']=np.array(ro['vel'][:,0])
	d['vy']=np.array(ro['vel'][:,1])
	d['vz']=np.array(ro['vel'][:,2])
	d['part_id']=ro['part_id']
	d['popid']=np.array(ro['part_id'])
	d['lgc'],d['pzgc'],d['rgc']=autil.xyz2lzr(d['px'],d['py'],d['pz'])
	d['vlgc'],d['vzgc'],d['vrgc']=autil.vxyz2lzr(d['px'],d['py'],d['pz'],d['vx'],d['vy'],d['vz'])
	d['lgc']=d['lgc']%360.0
	d['pzgc_abs']=np.abs(d['pzgc'])
	d['vphi']=-(d['vlgc'])
	#........
	d['vlgc']=-(d['vlgc'].copy())	
	#........
	d['vphi']=-(d['vlgc'])
	d['tstamp'] = np.zeros(d['px'].size) + tstamp
	
	d['pxgc'], d['pygc'] = d['px'],d['py']
	del d['px']
	del d['py']
	
	
	
	print('saving only thick and thin disc stars...')
	tabpy.where(d,(d['part_id']==5)|(d['part_id']==7))	
	
	if clock:
		timeit.end()
	return d

def getsims():
	timeit = stopwatch()
	timeit.begin()	
	

	simname = 'agama_test_hires'	
	svloc = '/import/photon1/skha2680/Documents/phd_work/tepper_sims/warp/'+simname
	os.mkdir(svloc)
	
	import natsort
	from natsort import natsorted, ns
	dloc = '/export/photon1/tepper/ramses_output/Nbody/agama/myics/'+simname				
	filenames = os.listdir(dloc)
	filenames = natsorted(filenames)
	filenames = np.array(filenames)	
	
	for fl in filenames:
		if fl.startswith('output_00140'):
			print(fl)
			dt = read_tepper_pynbody(dloc+'/'+fl,clock=False)			
			ebfwrite(dt,fl,svloc)		

	timeit.end()
	return 
		
		

# Miscellaneous			


def points_sphere_equ(N=100,r=10,smpfac=0.1):
	
	'''
	returns equidistant points on a sphere
	
	https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf	
	
	
	N = number of points to generate
	r = radius at which to generate
	

	# dts = points_sphere_equ(N=1000,r=50)
	# # dtools.picklewrite(dts,'pointings',desktop)
	
	# dts = dtools.pickleread(desktop+'/pointings.pkl')	
	
	'''


	Nc = 0		
	r1 = 1.
	a = (4*np.pi*(r1**2.))/N
	d = np.sqrt(a)
	mtheta = int(np.round(np.pi/d))
	dtheta= np.pi/mtheta
	dphi = a/dtheta
	dts = {}
	dts['xv'] = []
	dts['yv'] = []
	dts['zv'] = []
	for m in range(0,mtheta-1):
		theta = (np.pi*(m + 0.5))/mtheta
		mphi = int(np.round(2*np.pi*(np.sin(theta)/dphi)))
		for n in range(0,mphi-1):
			phi = (2*np.pi*n)/mphi
			x = r*np.sin(theta)*np.cos(phi)
			y = r*np.sin(theta)*np.sin(phi)
			z = r*np.cos(theta)
			
			dts['xv'].append(x)
			dts['yv'].append(y)
			dts['zv'].append(z)
			
	for ky in dts.keys():
		dts[ky] = np.array(dts[ky])		
		
	dts['l'],dts['b'],dts['d'] = autil.xyz2lbr(dts['xv'],dts['yv'],dts['zv'])
	dts['l']=dts['l']%360.	
	
	indall = np.arange(0,dts['l'].size)
	np.random.shuffle(indall)
	# print(smpfac)
	smpsize = int(smpfac*indall.size)
	# print(smpsize)
	dts = dtools.filter_dict(dts,indall[:smpsize])
	
	dts['dgrid'] = np.linspace(0.,30,100)
	
	return dts



def sech2(x):
    """
    sech squared distribution

    Arguments:
        x = flot, int or array

    Returns:
        sech2(x), same format as x

    2019-01-19
    """
    return 1./np.cosh(x)**2
    
    
def tcalc(totsize,chunksize,tperchunk):
	
	'''
	totsize: total number of objects to run
	chunksize: total per chunk
	
	tperchunk : seconds
	
	'''
	
	tmp1 = float(totsize)/float(chunksize)
	
	nhrs = ((tperchunk*tmp1)/3600.)
	ndays = nhrs/24.
	
	

	return str(nhrs)+' hours', str(ndays)+' days' 

	
def fcount(floc,flist=False,nlist=False):
	
	'''
    NAME: fcount

    PURPOSE: counts the number of files in a given directory


    INPUT: file location 

    OUTPUT: number count 
       
    HISTORY: October 27, 2022 (INAF Torino)
    	
	'''
	
	
	cnt = []
	
	for fl in os.listdir(floc):
		cnt.append(fl)
		
	cnt = np.array(cnt)	
	
	
	

	print(str(cnt.size)+' files in total')
	
	if flist:
		# os.system('ls -lh '+floc)	
		return cnt
	elif nlist:
		return cnt.size	
	else:
		os.system('ls -lh '+floc)	
		return 


def getphi_(glon,typ=''):

	'''
	'' type for overplotting glon on healpix
	'2' for normal glon, but with glon > 180 as negative
	'''

	if typ == '':
		phival = glon.copy()
		indg = np.where(glon < 180)[0]
		phival[indg] = -(glon[indg]).copy()
		indl = np.where(glon > 180)[0]
		phival[indl] = 360. - glon[indl].copy()			

	if typ == '2':
	
		phival = glon.copy()
		indg = np.where(glon < 180)[0]
		phival[indg] = (glon[indg]).copy()
		indl = np.where(glon > 180)[0]
		phival[indl] =  glon[indl].copy() - 360	
	
	return phival
																										
def comments_form():
	
	'''
		
    NAME:

    PURPOSE:


    INPUT:

    OUTPUT:
       
    HISTORY:        
	
	'''	


	'''
		
    NAME: Einasto density 

    PURPOSE: evaluate Einasto density at a given radius (or rq)
	

    INPUT: 
    
	rv = radius at which to compute (default=1)
	ntot = Total number/mass of stars in volume (default=1)
	n = Einasto profile shape (default=1)
	rs = half-mass radius (default=1) 
	p = triaxiality (y/x, default=1)
	q = triaxiality (z/x, default=1)
	
    OUTPUT: density (rgcd)
    
    REMARKS:
    
    -analytical form taken from Merrit 2006 (page 15)
    -also refer to Retana-Montenegro 2012 for expressions
       
    HISTORY:   
    
    28 September 2020 (RUG)
    	
	'''


	
	#Java:

	'''
		
	/**
	 * NAME: 
	 * PURPOSE:
     * INPUT:
 	 * OUTPUT:
     * HISTORY:        
	
	 */
	
	
	
	
	
	'''	
	
	return 


class cltemplate(object):

	'''
	empty class template

	'''	
	def __init__(self,data=None):
		print('')
		
	def fnc(self):
		
		
		return 


		
def ncombs(N,ncmb=2):
	'''
	No. of unique combinations
	'''
	print('N = '+str(N))
	print('unique combinations of '+str(ncmb))
	
	num1 = float(np.math.factorial(N))
	num2 = float(np.math.factorial(N-ncmb))
	num3 = float(np.math.factorial(ncmb))
	
	num1/(num2*num3)
	return num1/(num2*num3)


def tpass(n=10,t=5):
	'''
	dummy run 
	'''
	
	for i in range(n):
		print(i)
		time.sleep(t)
		
def dbug():
	
	print('--------------------------')
	print('testing..............')

	print('--------------------------')
	
	return

def basic_pool_example():
	




	exec(open("./packages.py").read()) 
	
	desktop = dtools.getdirec('desktop')
	
	os.environ["NUMEXPR_MAX_THREADS"] = "5"
	
	
	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--ncpu',nargs='+')
	parser.add_argument('--flen',nargs='+')
	args = parser.parse_args()
	ncpu= int(args.ncpu[0])
	flen= int(args.flen[0])
	
	
	timeit = dtools.stopwatch()
	timeit.begin()
	
	
	
	stp1 = False
	
	
	# notes: send this file to cluster - > desktop+'/xgbrc.ebf'
	# notes: send this directory to cluster - > desktop+'/dummy/'
	
	
	if stp1:
		
		'''
		creates files
		'''
		fl = tabpy.read(desktop+'/xgbrc.ebf')	
		fl = dtools.mk_mini(fl,fac=0.06,key='source_id')
		
		os.system('rm -rf '+desktop+'/dummy')
		os.system('mkdir '+desktop+'/dummy')	
		
		for inum in range(flen):
			print('running...'+str(inum))
			# fl = tabpy.read(desktop+'/dummy.fits')
					
			val1 = np.random.uniform(0.,100000,1)
			val = val1**2. + fl['source_id']
			dtools.picklewrite(val,'test_'+str(inum),desktop+'/dummy')
			
	
	def myfunc(i,floc):
	
	
		fl = dtools.pickleread(floc+'/test_'+str(i)+'.pkl')
		res = {}
		
		res['source_id'] = fl
		res['value'] = (fl + fl) + np.sin(0.4)
		
		
		
		
		return res	
	
	
	def stp2(ncpu):
		
		
		'''
		uses pool
		'''
		
		
		floc = desktop+'/dummy'
		nfiles = dtools.fcount(floc,nlist=True)
		indres = np.arange(nfiles)
		from multiprocessing import Pool
		
		print('all fine')
		
		p=Pool(ncpu)
		data_postdist = p.map(partial(myfunc,floc=floc), indres)
		data_pd={}
		for key in data_postdist[0].keys():			
			data_pd[key]=np.concatenate([d[key] for d in data_postdist]).transpose()	
				
	
		p.close()	
		
		dtools.fitswrite(data_pd,'rubbish',desktop)
		
	
		
		return
		
		
	stp2(ncpu)
	timeit.end()	
		
	return 	


















