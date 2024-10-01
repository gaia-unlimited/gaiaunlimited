'''
# one stop script for all red clump related tools
'''

exec(open("./packages.py").read()) 

# imp.reload(dtools)

timeit = dtools.stopwatch()
timeit.begin()


#.............................
#.............................

zsolar=0.019


#.............................
#.............................


def rc_distfunc(dist,sigabsmag,sigplx=0,dplx=100):
	
	'''
	estimate distance uncertainty in RC
	

	returns the distance error from inverse parallax
	'''
	
	sig_d_rc = (np.log(10)*abs(sigabsmag))*dist*0.2			
				
	sig_d_plx = (sigplx*(dplx**2.))		
	
	
	return sig_d_rc, sig_d_plx
	
def rc_distfunc_interp(d,band='g'):
	
	'''
	temporary: interpolate distance uncertainty from d, based on the curve for w1 or g
	returns: sig_drc(w1 or g)
	
	'''
	
	root_ = '/iranet/users/pleia14//Documents/pdoc_work/science/edr3/rc/calrcdir'
	duncprof = tabpy.read(root_+'/duncprof.ebf')
	

	if band == 'g':
		print('grabbing from g')
		dgrid = duncprof['dg']
		sigdgrid = duncprof['sigdg']
	if band == 'w1':
		print('grabbing from w1')
		dgrid = duncprof['dw1']
		sigdgrid = duncprof['sigdw1']

	from scipy import interpolate
	f = interpolate.interp1d(dgrid,sigdgrid,bounds_error=False,fill_value=(sigdgrid[0],sigdgrid[-1]))
	sigd = f(d)
			
	return sigd

def jk0_from_teff_feh(teff,feh):
	
	'''
	gets intrinsic colour 2mass, from teff, feh using fits from khanna et al. 2018
	
	'''

	glxdir = dtools.getdirec('galaxia')
	calib_file = 'k18rc_jk_teff_feh_coeff.pkl'


	xs = np.array([feh,teff])		

	popt_pop = dtools.pickleread(glxdir+'/'+calib_file)
	popt = np.array([popt_pop[1],popt_pop[2],popt_pop[3],popt_pop[4],popt_pop[5],popt_pop[6]])
	popt = popt.astype(float)	

	clr = (func(xs,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]))		
	
	return clr


def alambda_rc(j,ks,clr_jk0,flambda):
	
	'''
	get A_lambda for red clump like stars, using the RJCE method from Khanna et al. 2018
	'''
	
	fj = dtools.redden_fac()['jmag']	
	fk = dtools.redden_fac()['kmag']	

	alambda = flambda*((j-ks)-clr_jk0)/(fj-fk)
	
	return alambda
	

def mks_feh(feh_dt):
	
	'''
	return mks from feh
	
	'''
	
	feh = np.linspace(-0.8,0.4,13)
	mks = np.array([-1.39,-1.405,-1.442,-1.475,-1.520,-1.564,-1.598,-1.622,-1.633,-1.646,-1.659,-1.666,-1.676])	
	
	mk_dt=np.interp(feh_dt,feh,mks)	
	
	return mk_dt
	


def myselfunc_rc(mydata,zcuts=False,rem_rogue=False,tstcut=False,galaxia=False):
	
	print('')
	
	if 'clr' not in mydata.keys():
			
	
		mydata['Z'] = (10**(mydata['feh']))*zsolar
	
		met = mydata['feh'] 
		teff = mydata['teff']
		
		mydata['clr'] = jk0_from_teff_feh(mydata['teff'],mydata['feh'])
	
	############################################
	## SELECT RED-CLUMPS
	############################################		
	teff_ref = -(382.5*mydata['feh']) + 4607
	loggup = 0.0018*(mydata['teff'] - teff_ref) + 2.5	
	zlower = 1.21*((mydata['clr'] -0.05)**9.0) + 0.0011
	zupper = 2.58*((mydata['clr'] -0.40)**3.0) + 0.0034			
			
	teffmax = 5383
	teffmin = 4227
	cond1 = (mydata['teff'] > teffmin)&(mydata['teff'] < teffmax)
	#cond1 = (mydata['clr'] > 0.5)&(mydata['clr'] < 0.8)		
	cond2 = (mydata['logg']>= 1.8)&(mydata['logg']<= loggup)
	cond3 = 1
	cond4 = 1#(mydata['absks']> -1000052)						
	cond5 = 1

	if zcuts:			
		cond3 = (mydata['Z']<= 0.06)&(mydata['Z']>=zlower)&(mydata['Z']<=zupper)			

	if rem_rogue:
		print('removing rogue..')
		cond4 = (mydata['absks']> -2.0)


	if tstcut:
		cond2 = (mydata['logg']>= 2.3)&(mydata['logg']<= loggup)
	
	if galaxia:
		cond5 = (mydata['smass'])>mydata['mtip']
		
	conds = cond1&cond2&cond3&cond4&cond5	
	
	
	return conds,zlower,zupper,teff_ref,loggup


def myselfunc_rgb(mydata,loggkey='',teffkey=''):
	
	'''
	basic selection based on dr3 gaia paper
	
	'''
	
	if loggkey == '':
		loggkey = 'logg'
	if teffkey == '':
		teffkey = 'teff'
	
	cond1 = mydata[loggkey] < 3.5
	cond2 = (mydata[teffkey] > 3000)&(mydata[teffkey] < 5500) 
	conds = cond1&cond2 
	
	
	
	return conds



############################################
## CALIBRATION FUNCTIONS (f(Fe/H, (J-K)_0))
############################################
def func(xs,a0,a1,a2,a3,a4,a5,prnt=False):
	# Z = (j-k)_0, xs = [feh,Teff]
	Z_key = 'jk0'; X_key = '[FeH]'; Y_key = '(5040/Teff)'
	X = xs[0]
	Y = 5040.0/xs[1]
	Z = a0 + a1*X + a2*(X**2.0) + a3*X*Y + a4*Y + a5*(Y**2.0)			
	if prnt:
		a0 = np.round(a0,3);a1 = np.round(a1,3);a2 = np.round(a2,3);a3 = np.round(a3,3);a4 = np.round(a4,3);a5 = np.round(a5,3)
		print(Z_key+' = '+str(a0)+'+ '+str(a1)+X_key+'+ '+str(a2)+'('+X_key+'^2)'+'+ '+str(a3)+X_key+Y_key+' + '+str(a4)+Y_key+'+ '+str(a5)+'('+Y_key+'^2)')
	return  Z
def transpose_table(tab_before, id_col_name='ID'):   
	colnms = np.linspace(0,len(tab_before)-1,len(tab_before))
	colnms = colnms.astype('string')
	new_colnames=tuple(colnms)
	
	new_rownames=tab_before.colnames[:]
	tab_after=Table(names=new_colnames)
	for r in new_rownames:
		tab_after.add_row(tab_before[r])
	if id_col_name != '':
		tab_after.add_column(Column(new_rownames, name=id_col_name),index=0)
	return(tab_after)
def sv_res(pop,popt,sz):
	############################################
	## WRITE RESULTS
	############################################	
	pltdir = caldir	
	calib_file = 'calib_res_direct.ebf'
	print(popt)
	
	if os.path.isfile(pltdir+'/'+calib_file):		
		print('updating '+calib_file)
		results = ebf.read(pltdir+'/'+'calib_res_direct.ebf')				
	else:
		print('Creating '+calib_file) 
		results = {}
	
	# ex:- [dwarfs,a0,a1,a2.....]		
	popt_pop = list(popt); popt_pop.insert(0,pop); popt_pop = np.array(popt_pop)
	results[pop] = popt_pop			
	ebf.write(pltdir+'/'+calib_file,'/',results,'w')	
	
	#mk Table
	tab = Table(results)
	tab.write(pltdir+'/'+calib_file+'table',format='latex')
	return


############################################
## SELECT POPULATION (DWARFS/GIANTS/CLUMPS)
############################################
def sel_pop(mydata,pop):
	
	print('Total '+str(mydata['teff'].size)+' stars')

	##########################################
	## REMOVE M-DWARFS (4200 < Teff < 8000 K) 
	##########################################
	ind = np.where((mydata['teff'] > 4200.0)&(mydata['teff'] < 8000.0))[0]		
	print('REMOVING M-DWARFS ->  select 4200 < Teff < 8000 K')
	mydata = sutil.filter_dict(mydata,ind)		
	print('Total '+str(mydata['teff'].size)+' stars selected')

	#############################################
	## Choose population (dwarfs/giants/clumps)
	#############################################

	if pop == 'dwarfs':
		ind = np.where((mydata['logg'] >= 3.8)&(mydata['smass']<mydata['mtip']))[0]		
	if pop == 'giants':
		ind = np.where((mydata['logg'] <= 3.2)&(mydata['smass']<mydata['mtip']))[0]		
	if pop == 'clumps':		
		ind = np.where((mydata['logg'] >= 1.8)&(mydata['logg'] <= 3.0)&(mydata['smass']>mydata['mtip'])&(mydata['absks']>-2.0))[0]
#		ind = np.where((mydata['logg'] <= 3.0)&(mydata['smass']>mydata['mtip']))[0]

	if pop == 'giants_comb':		
		ind = np.where((mydata['logg'] <= 3.8))[0]

	mydata = sutil.filter_dict(mydata,ind)	
	print('selected '+str(mydata['teff'].size)+' '+pop+' stars')
	
		
	return mydata

def rd_it(zsolar=0.019,iso='tmasswise',errors=False):			
	loc = datadir
	if errors:
		mydata = ebf.read(loc+'/minifile_errors.ebf')	
	else:
		mydata = ebf.read(loc+'/minifile.ebf')
		
	mydata['teff'] = 10**mydata['teff']
	mydata['logg'] = mydata['grav']
	mydata['Z'] = (10**(mydata['feh']))*zsolar
	print('using '+iso+'  photometry')
	mydata['absj'],mydata['absks'] = mydata[iso+'_j1'],mydata[iso+'_ks1']		
	mydata['ajk0'] = mydata['absj'] - mydata['absks']					
	
	
	#print('')
	#print('Adding mtip_new')
	#mydata['log_age'] = mydata['age']
	#loc = HOME+'/GalaxiaWork'
	#d = ebf.read(loc+'/feh_age_mtip.ebf','/')
	#sgrid=sutil.InterpGrid(tabpy.npstruct(d),['feh','log_age'])
	#mydata['mtip_new'] = sgrid.get_values('mtip',[mydata['feh'],mydata['log_age']],fill_value='nearest',method='nearest')	
	mydata['mtip_new'] = mydata['mtip']
		
	return mydata	


class plt_rc_sel(object):		
	
	def __init__(self,mydata,zsolar=0.019,iso='tmasswise',use_res='y'):		
		self.pop = 'clumps'
		self.mydata = mydata
		self.zsolar = zsolar
		self.iso = iso
		self.use_res = use_res

		self.loc = datadir

		self.calib_file = 'calib_res_direct.ebf'
		# self.caldir = caldir		
#		self.caldir = '/home/shourya/Documents/phd_work/GalaxiaWork/calibrations/now_direct'		
	
	def read_results(self):
		
		
		mydata = self.mydata
		# pop = self.pop
		# print('reading results for '+self.pop) 	
		# xs = np.array([mydata['feh'],mydata['teff']])			
		# results = ebf.read(self.caldir+'/'+self.calib_file)
		# print(results)
		# popt_pop = results[pop]
		# popt = np.array([popt_pop[1],popt_pop[2],popt_pop[3],popt_pop[4],popt_pop[5],popt_pop[6]])
		# popt = popt.astype(np.float)
		# self.popt= popt	
		
		glxdir = dtools.getdirec('galaxia')
		calib_file = 'k18rc_jk_teff_feh_coeff.pkl'
		

		# popt_pop = ebf.read(HOME+'/GalaxiaWork/calibrations/now_direct/calib_res_direct.ebf')['clumps']
		popt_pop = dtools.pickleread(glxdir+'/'+calib_file)
		popt = np.array([popt_pop[1],popt_pop[2],popt_pop[3],popt_pop[4],popt_pop[5],popt_pop[6]])
		self.popt = popt.astype(np.float)
		
	def sel_clumps(self,mydata,zcuts,use_true=False,simple_sel=False,rem_rogue=False,tstcut=False,mtip_to_use='mtip_new'):		
			
		popt = self.popt
		met = mydata['feh'] 
		teff = mydata['teff']
		xs = np.array([met,teff])	
	


		if use_true:
			print('')
			print('USING TRUE COLORS')
			mydata['clr'] = mydata['tmasswise_j1']-mydata['tmasswise_ks1']

		else:
			print('')
			print('USING DERIVED COLORS')			
			mydata['clr'] = (func(xs,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]))
			

		print('')
		print('Using '+mtip_to_use)
		print('')
		ratio = mydata['smass']/mydata[mtip_to_use]			

		conds,zlower,zupper,teff_ref,loggup = myselfunc_rc(mydata,zcuts=zcuts,rem_rogue=False,tstcut=False)
		
		ind = np.where(conds)[0]	

		if simple_sel:		
			print('')
			print('USING SIMPLE CUTS')
			ind = np.where((ratio > 1.0)&(mydata['logg']>= 1.8)&(mydata['logg']<= 3.0)&(mydata['teff'] > teffmin)&(mydata['teff'] < teffmax)&cond3&cond4)[0] 

			#-------------------------------------
			#ind = np.where((ratio > 1.0)&cond3&cond4)[0] 			
			#ind = np.where((ratio > 1.0)&(mydata['logg']>= 1.8)&(mydata['logg']<= 3.5)&cond3&cond4)[0] 			
			#ind = np.where((ratio > 1.0)&(mydata['logg']>= 1.8)&(mydata['logg']<= 3.5)&(mydata['teff'] > teffmin)&(mydata['teff'] < teffmax)&cond3&cond4)[0] 			
			#ind = np.where((ratio > 1.0)&(mydata['logg']>= 1.8)&(mydata['logg']<= 3.5)&(mydata['teff'] > teffmin)&(mydata['teff'] < teffmax)&cond3&cond4)[0] 						  
			#ind = np.where((ratio > 1.0)&(mydata['logg']>= 2.6)&(mydata['logg']<= 3.0)&(mydata['teff'] > teffmin)&(mydata['teff'] < 5000)&cond3&cond4)[0] 
					
		mydata['zlower'],mydata['zupper'] = zlower,zupper
		mydata['teff_ref'], mydata['loggup'] = teff_ref, loggup
		mydata = dtools.filter_dict(mydata,ind)		
																																																																			
		print('')																																																																																			
		print('Teff stats')
		print(autil.stat(mydata['teff']))
		return mydata
	def interpolate(self):
		'''
		[FeH] vs. Mks
		'''	
		mydata = self.mydata
		iso = self.iso
		loc = self.loc
		
		if self.mk_interp=='y':
			print('saving interp file')
			
			dtools.profile(mydata['feh'],mydata['absks'],range=[-1.,1.5],bins=100,label='',mincount=5,func=np.median,lw=4,return_profile=True)			
			mydata = sutil.runavg_x_y(mydata,'feh','absks',100)
			absks_feh_intp = {'absks_intp':mydata['absks_intp'], 'feh_intp':mydata['feh']}
			ebf.write(loc+'/absks_feh_intp_'+iso+'_direct.ebf','/',absks_feh_intp,'w')
	
		
		
		return

	def plt_purity(self):
		ratio = self.ratio
		above_per = self.above_per
		err = self.err
		
		plt.hist(ratio,bins=100,histtype='step',normed=True,label=str(above_per)+' % ('+err+')')				
		plt.xlim([0.99,1.02]) 
		plt.ylim([0,260])	
		plt.axvline(1.0,linestyle='--',color='black')			
#		sutil.pltLabels(r'$\frac{Mass}{Mass_{\mathrm{Tip}}}$','Density')
		sutil.pltLabels('Mass/ Tipping Mass','Density')		
		
		return

	def plt_feh_mks(self,plm,lbl=True,cb=False):
			
		mydata = self.mydata
		feh_pnts = self.feh_pnts
		model_absks_intp = self.model_absks_intp

		
		h = sutil.hist_nd([mydata['feh'],mydata['absks']],bins=100);
		
		im1 = h.imshow(cmapname='Greys',norm=matplotlib.colors.LogNorm(),smooth=False)	
		plt.plot(mydata['feh'],mydata['absks_intp'],'r.',markersize=0.1)		
		
		plt.xlim([-0.74,0.4])
		plt.ylim([-2.0,-1.0])		
		
		#ax=plm.axes
		#im1 = h.imshow(cmapname='Greys',norm=matplotlib.colors.LogNorm(),ax=ax,smooth=False)			
		#ax.plot(feh_pnts,model_absks_intp,'red')						
		#ax.set_xlabel('[Fe/H]')
		#ax.set_xlim([-0.74,0.4])
		#ax.set_ylim([-2.0,-1.0])		
				
		if lbl:
			#ax.set_ylabel('M$_{\mathrm{Ks}}$')
			plt.ylabel('M$_k$')		
			plt.xlabel('[Fe/H]')		
		if cb:
			plt.colorbar(pad=0.0,label='Density')	



##		plt.text(0.05,0.8,err,color='black',fontweight='bold',transform=plt.gca().transAxes)	

		##err = self.err
		##plt.title(err,color='black',fontweight='bold')	
		#plt.xlim([h.locations[0][0],h.locations[0][-1]])		
		


		
		return im1

	def plt_clr_Z(self,plm,lbl=True,cb=True,typ=''):		
		mydata = self.mydata
		zlower_mod = self.zlower_mod
		zupper_mod = self.zupper_mod
		clrs = self.clrs
		plm.next()
		xrng = [0.5,0.8]
		yrng = [0.,0.04]
		deltax = 0.0027
		deltay = 0.0005
		
		xbins = int((xrng[1] - xrng[0])/deltax)
		ybins = int((yrng[1] - yrng[0])/deltay)

		print('')
		print('Xbins, Ybins:')
		print(xbins, ybins)
		print('')		
		
		h = sutil.hist_nd([mydata['clr'],mydata['Z']],bins=[xbins,ybins],normed=True,range=[xrng,yrng])
		tmp = h.apply(mydata['absks'],np.median)
		ax1 = plm.ax
		if typ =='dens':
			tmp = h.apply(mydata['absks'],len)			
			ind=np.where(h.data<1.)[0]
			tmp[ind] = np.nan
			im = h.imshow(tmp,cmapname='inferno',smooth=False,norm=matplotlib.colors.LogNorm(),ax=ax1)			

		else:
			
			ind=np.where(h.data<1.)[0]
			tmp[ind] = np.nan
			levels= (-1.8,-1.75,-1.7,-1.65,-1.6,-1.55,-1.5)		
			
			print('....checking here...')
			
			print(tmp)	
			print('......')
			im = h.imshow(tmp,cmapname='jet',vmin=-1.8,vmax=-1.5,smooth=False,ax=ax1)#,norm=matplotlib.colors.LogNorm())
#			h.contourf(tmp,cmapname='jet',vmin=-1.8,vmax=-1.5,smooth=True,levels=levels) #norm=matplotlib.colors.LogNorm()
#			h.imshow(tmp,cmapname='jet') #norm=matplotlib.colors.LogNorm()

		
		if lbl:
			plt.ylabel('$Z$')

		if cb:
			plt.colorbar(pad=0.0)	
#			plt.colorbar(pad=0.0,use_gridspec=False,location='top')	
#			plt.colorbar(use_gridspec=False,location='top')	
	
#		plt.plot(mydata['clr'],mydata['zlower'],'k.')
#		plt.plot(mydata['clr'],mydata['zupper'],'k.')				

		lw=2.5
		CLR = 'cyan'
		ax = plm.ax		
		ax.plot(clrs,zlower_mod,COLOR=CLR,linestyle='--',linewidth=lw)
		ax.plot(clrs,zupper_mod,color=CLR,linestyle='--',linewidth=lw)			
		
		ax.set_ylim([h.locations[1][0],h.locations[1][-1]])
		ax.set_xlim([h.locations[0][0],h.locations[0][-1]])	
		
		ax.set_ylim([0.002,0.03])		
		ax.set_xlim([0.52,0.75])
		ax.set_xlabel('$(J - K)_{0}$')
		ax.set_ylabel('$Z$')	
		return im

	def plt_mks_diff(self,lbl=True,clr='red'):
		mydata = self.mydata
		

		mn_del_Mk = np.round(np.median(mydata['absks']-mydata['absks_intp']),2)
		sig_del_Mk = 0.5*(np.percentile(mydata['absks']-mydata['absks_intp'],84)-np.percentile(mydata['absks']-mydata['absks_intp'],16))
		sig_del_Mk = np.round(sig_del_Mk,2)
		plt.hist(mydata['absks']-mydata['absks_intp'],bins=100,histtype='step',normed=True,label='<$ \Delta M_{\mathrm{k}}$>='+str(mn_del_Mk)+'$\pm$'+str(sig_del_Mk),color=clr)		
	
		plt.axvline(0,ymax=0.85,color='black',linestyle='--')	
		
		plt.xlim([-0.5,1.0])
		
		if lbl:
			plt.ylabel('Density')						


		plt.xlabel('M$_{\mathrm{k}}$ - M$_{\mathrm{k}}$(pred) ')

			
						
		plt.ylim([0,6.5])

#		plt.savefig(os.path.join(pltdir,"{}.png".format('clumps_sel')))			
		return	mn_del_Mk,sig_del_Mk 

	def get_dist(self,dist_method='dmod',Mk=-1.65):	

		
		data = self.mydata		
		
		print('method = '+dist_method)	
		if dist_method == 'dmod':
			dmod = autil.dist2dmod(data['rad'])
		
		elif dist_method == 'rc':
			# read Mk-[Fe/H] curve
			print('reading Mks_[Fe/H] interp file')
			loc = self.loc			
			nam_intp_file = 'absks_feh_intp_tmasswise_direct'			
			absks_feh_intp = ebf.read(loc+'/'+nam_intp_file+'.ebf')
			f = interp1d(absks_feh_intp['feh_intp'],absks_feh_intp['absks_intp'])	
		
			ind = np.where((data['feh'] > min(absks_feh_intp['feh_intp']))&(data['feh'] < max(absks_feh_intp['feh_intp'])))[0]	
			data = sutil.filter_dict(data,ind)
			data['absks_intp'] = f(data['feh'])
			Mk = data['absks_intp']			
			
			# extinction correction
			'''   f(i) = A(i)/E(B-V); use GALAXIA tables --> UKIRT J & K        ''' 
			fj = 0.902
			fk = 0.367
			data['Ak'] = (fk/(fj-fk))*(data['tmasswise_j']-data['tmasswise_ks'] - data['clr']); print('using new Ak correction')	
			dmod = data['tmasswise_ks']- Mk - data['Ak']
			
			
		data['d'] = (10.0*(np.power(10,((dmod)/5.0))))/1000 # in kpc
		data['dmod_used'] = autil.distmod(data['d'])
		
		self.mydata = data		
		return 	
		
		
	def run(self,lbl=True,errors='n',mk_interp='n',zcuts=True,use_true=False,simple_sel=False,rem_rogue=False,tstcut=False,mtip_to_use='mtip_new'):		

		self.mk_interp = mk_interp
		loc = self.loc
		iso = self.iso

		self.read_results()	
		popt = self.popt			
		
		
		if errors =='n':		
			self.err= ''		
		elif errors == 'y':		
			self.err='+Spectroscopic errors'		
		self.err = self.err.upper()


		self.mydata = self.sel_clumps(self.mydata,zcuts=zcuts,use_true=use_true,simple_sel=simple_sel,rem_rogue=rem_rogue,tstcut=tstcut,mtip_to_use=mtip_to_use)			
		mydata = self.mydata	
		print(str(mydata['px'].size)+' clumps selected')	



		ratio = mydata['smass']/mydata[mtip_to_use]
		above = (np.where(ratio > 1.0)[0])
		below = (np.where(ratio < 1.0)[0])
		above_per = (float(len(above))/mydata['smass'].size)*100.0; above_per = np.round(above_per,1)
		print(str(above_per)+' % purity ')		
		print(str(len(above))+' No. of pure ')		
		self.ratio = ratio
		self.above_per = above_per		
		are_above = mydata['fid'][above]
		self.mydatarc = mydata

		#----------
		xs = np.array([mydata['feh'],mydata['teff']])
		mydata['clr']= (func(xs,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],prnt=True))	
		mn= min(mydata['clr'])
		mx = max(mydata['clr'])
		self.clrs = np.linspace(mn,mx,mydata['clr'].size)
		
		self.zlower_mod = 1.21*((self.clrs -0.05)**9.0) + 0.0011
		self.zupper_mod = 2.58*((self.clrs -0.40)**3.0) + 0.0034			
		
		self.interpolate()
		# print('reading interp file')
		# absks_feh_intp = ebf.read(loc+'/absks_feh_intp_'+iso+'_direct.ebf')
		# f = interp1d(absks_feh_intp['feh_intp'],absks_feh_intp['absks_intp'])
	
	
		# ind = np.where((mydata['feh'] > min(absks_feh_intp['feh_intp']))&(mydata['feh'] < max(absks_feh_intp['feh_intp'])))[0]	
		# mydata = sutil.filter_dict(mydata,ind)
		# mydata['absks_intp'] = f(mydata['feh'])
		# self.mydata = mydata
		
		# feh_mn = min(mydata['feh'])
		# feh_mx = max(mydata['feh'])
		# self.feh_pnts = np.linspace(feh_mn,feh_mx,100)
		# self.model_absks_intp = f(self.feh_pnts)

		#-----------------------------

					

		return are_above


