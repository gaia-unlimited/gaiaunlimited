

exec(open("./rcdemo/packages_to_import.py").read()) 



######### density functions





def exp_norm_bessel(rd,rcut,print_=False):

	'''
	normalisation for exp disc with rcut [integrated 0 to infinity], uses bessel function second kind
	'''

	import scipy.special as sc
	

	tmp1 = (2.*rcut*rd)
	tmp2 = sc.kn(2,2.*(np.sqrt(rcut))/(np.sqrt(rd)))
	tmp3 = (tmp1*tmp2)
	
	if print_:
		print(tmp1)
		print(tmp2)
		print(tmp3)
	
	return tmp3

def expdisc(xyz,rq=0,zq=0,Rd=2,l=0,b=0,d=0,hz=0.5,xyzmode=True,lbdmode=False,lbdmode_gal=False,R0=8.275,Rflare_=1.12,mass=1.,return_surfdens=False,innerCutoffRadius =0.,
			nsersic = 1.,
			Rmax=np.inf,phiw = 120,Rw = 8.,aw = 1.,hw0 = 0.1,warp='off',xsun=-8.275):

	'''
	here rq = R
		# Rsun = -dtools.get_lsr()['xsun']
		# Rflare = Rflare_*Rsun
	'''
	
	if xyzmode:
		pxgc = xyz[:,0]
		pygc = xyz[:,1]
		zq = xyz[:,2]
		rq = np.sqrt(pxgc**2. + pygc**2.)		
		
	if lbdmode:

		x,pygc,zq = autil.lbr2xyz(l,b,d)
		pxgc = x+xsun 		
		rq = np.sqrt(pxgc**2. + pygc**2.)

	if lbdmode_gal:

		pxgc,pygc,zq = autil.lbr2xyz(l,b,d)
		rq = np.sqrt(pxgc**2. + pygc**2.)
			
	# Rflare = Rflare_*R0	
	Rflare = Rflare_#*R0	
	hz_ = hz*np.exp((rq - R0)/Rflare)	
	
	

	### eloisa version
	# # # # # zw = 0
	# # # # # if warp == 'on':

		# # # # # phiradcells =np.arctan(-pygc/-pxgc)  # fix this later (currently only works with xyzmode=True)
							
		# # # # # hw=np.zeros(len(rq))
		# # # # # i_in_warp=numpy.where(rq > Rw)
		# # # # # hw[i_in_warp]= hw0*np.power(rq[i_in_warp] - Rw, aw)
		# # # # # zw = hw * numpy.sin(phiradcells - np.radians(phiw))
	zw = 0
	if warp == 'on':

		# phiradcells =np.arctan(-pygc/-pxgc)  # fix this later (currently only works with xyzmode=True)
														
							
		hw=np.zeros(len(rq))
		i_in_warp=numpy.where(rq > Rw)
		hw[i_in_warp]= hw0*np.power(rq[i_in_warp] - Rw, aw)
		

		phicells = np.degrees(np.arctan2(pygc,pxgc))%360. #myconvention
		zw = hw * numpy.sin(np.radians(phicells - phiw))*(-1)
		
		# print('using ep convention')
		# phiradcells = np.degrees(np.arctan2(pygc,pxgc))	#EP convention
		# zw = hw * numpy.sin(np.radians(phiradcells - phiw))			
		


	# for the no Rcut case	
	fac1 = (Rd**2.) - (Rd*(Rmax + Rd)*(np.exp(-Rmax/Rd)))
		
	if Rmax == np.inf:		
		fac1 = (Rd**2.)  
		if innerCutoffRadius > 0.:
			fac1 = exp_norm_bessel(Rd,innerCutoffRadius)
			
		
	norms_ = (mass/(2.*np.pi*fac1))	


	tmp1 = np.exp(-((rq/Rd)**(1./nsersic)) - (innerCutoffRadius/rq) )

	tmp2 = (1./(hz_*2.))*np.exp( -(abs(zq-zw)/hz_)) # correct
	

	# surface density (collapsed density along z)
	rho_surf = norms_*tmp1

	# # rho = tmp2*norms_*tmp1	 correct
	rho = tmp2*norms_*tmp1	
	

	if return_surfdens:
		return rho, rho_surf
	else:
		return rho
	






##### utility functions

																			
	
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
		


def crossmatch(src,quant,suffix='',prefix=''):
    """
    matches in src for each quant
    Arguments:
    quant- An array of items to cross match
    src  - An array of items with which to crossmatch
    Return:
    indl- Array giving index of src where match was found.
    indr- Array giving index of quant where match was found order same as indl.
    indrnm- Array giving index of quant where match was not found.
    
    """
    mydict=dict(zip(src,np.arange(src.size)))
    ind=np.zeros(quant.size,dtype='int64')
    if len(suffix+prefix)>0:
        for i,temp in enumerate(quant):
            ind[i]=mydict.get(prefix+temp+suffix,-1)
    else:
        for i,temp in enumerate(quant):
            ind[i]=mydict.get(temp,-1)
    indr=np.where(ind>=0)[0]
    indrnm=np.where(ind<0)[0]
    indl=ind[indr]
    return indl,indr,indrnm
    
    

def where(data1,condition):
    ind=np.where(condition)[0]
    select(data1,ind)


def select(data1,ind):
	for key in data1.keys():
		#print(key)
		data1[key]=data1[key][ind]

def fitsread(filename,ext=1):
	
	from astropy.io import fits    
	
	data1=np.array(fits.getdata(filename,ext))
	# data1=(fits.getdata(filename,ext))
	data={}
	for x in data1.dtype.names:
		data[x.lower()]=data1[x]
		
	return data

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

# plotting


def invert(axis):
	
	if axis == 'x':
		plt.gca().invert_xaxis()
	elif axis =='y':		
		plt.gca().invert_yaxis()
	elif axis == 'both':
		plt.gca().invert_xaxis()		
		plt.gca().invert_yaxis()		
	return



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


