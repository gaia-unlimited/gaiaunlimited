import matplotlib  as mpl
import matplotlib.pyplot  as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.ticker import MaxNLocator


exec(open("./rcdemo/packages_to_import.py").read()) 



def make_cmap(name):
	a={}
	a['red']=((0.0, 1, 1), (0.003, 1, 1), (0.003, 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5))
	a['blue']=((0.0, 1.0, 1.0),(0.003, 1.0, 1.0),(0.003, 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))
	a['green']=((0.0, 1.0, 1.0),(0.003, 1.0, 1.0),(0.003, 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0))
	newjet=LinearSegmentedColormap('newjet',a)        
	cm.register_cmap(cmap=newjet)
	
	
	
class Plm2(object):
	"""
	Uses object-oriented matplotlib 
	fig,axes=plt.subplot(rows,cols)    
	plm.ax is the current axis
	For sutil.hist_nd.imshow, you have to get back a mappable image object
	and pass it on to colorbar after tight_layout
	plm=putil.Plm2(2,2); 
	for in range(4):
		plm.next()
		im1=h.imshow(,ax=plm.ax)
		plm.next()
		im2=h.imshow(,ax=plm.ax)
	plm.tight_layout()
	For two colorbars on top
	plt.colorbar(im1,ax=plm.axes[:,0],location='top',pad=0,fraction=0.08)
	plt.colorbar(im2,ax=plm.axes[:,1],location='top',pad=0,fraction=0.08)
	For single colorbar on side
	plt.colorbar(im2,ax=plm.axes,location='right')
	"""

	def __init__(self,rows=1,cols=1,xsize=7.0,ysize=7.0,xmulti=False,ymulti=False,full=False,slabelx=0.89,slabely=0.1,slabel=True,order='left-right',slabclr='black'):
		#self.slabels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		self.slabels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ac','ad','ae','af','ag','ah','ai','aj']
		params = {'font.size':12,
				  'text.usetex':False,
				  'ytick.labelsize': 'medium',
				  'legend.fontsize': 'large',
				  'axes.linewidth': 1.0,
				  'xtick.labelsize': 'medium',
				  'font.family': 'sans-serif',
				  'axes.labelsize': 'medium'}
		mpl.rcParams.update(params)
		
		
		# if xsize>13.0:
		#     mpl.rcParams['font.size']=15
		#     mpl.rcParams['legend.fontsize']=15
		#mpl.rcParams['text.usetex'] = True
		if full:
			#mpl.rcParams['axes.titlesize']=8
			mpl.rcParams['font.size']=8
			mpl.rcParams['legend.fontsize']=8
		else: 
			mpl.rcParams['font.size']=15
			mpl.rcParams['legend.fontsize']=15
			
			#mpl.rcParams['font.size']=10
			#mpl.rcParams['legend.fontsize']=10            
		
		self.rows=rows
		self.cols=cols
		self.figno=0
		self.panelno=0
		self.xmulti=xmulti
		self.ymulti=ymulti
		self.slabelx=slabelx
		self.slabely=slabely
		self.slabel=slabel
		self.order=order
		self.fig,self.axes=plt.subplots(self.rows,self.cols,figsize=(xsize,ysize))
		self.slabclr = slabclr



	def panels(self):
		return self.rows*self.cols	


	def next(self,slabclr='black',weight='bold'): 		          
		self.slabclr = slabclr		
		self.weight = weight	
		self.figno+=1
		self.figno=(self.figno-1) % (self.rows*self.cols)+1
		if type(self.order) == list:
			self.panelno=self.order[self.figno-1]
		else:
			if self.order == 'top-down':
				r=(self.figno-1)/self.rows
				c=(self.figno-1)%self.rows
				self.panelno=int((c*self.cols+r)+1)                     # modified to int() by Shourya (feb 25 2021)
			else:
				self.panelno=self.figno
				

		self.ax=self.axes.flat[self.panelno-1]
		#---added by Shourya
		plt.sca(self.ax)
		self.lcount=0
		#---
		if (self.slabel) and (self.rows*self.cols > 1):
			self.ax.text(self.slabelx,self.slabely,'('+self.slabels[(self.panelno-1)%len(self.slabels)]+')',transform=self.ax.transAxes,color=self.slabclr,weight=self.weight)
	
		# if self.xmulti:
			# if (self.panelno-1)/self.cols != (self.rows-1):
				# self.ax.set_xticklabels('')
				# #ax.set_xticks([])
		if self.ymulti:
			if (self.panelno-1) % self.cols != 0:
				self.ax.set_yticklabels('')
				#ax.set_yticks([])

	def xlabel(self,s):        
		if self.xmulti:
			if (self.panelno-1)/self.cols == (self.rows-1):
				self.ax.set_xlabel(s)
		else:
			self.ax.set_xlabel(s)
	
	def ylabel(self,s):        
		if self.ymulti:
			if (self.panelno-1) % self.cols == 0:
				self.ax.set_ylabel(s)
		else:
			self.ax.set_ylabel(s)

	# def colorbar(self,*args,**kwargs):
	#     self.fig.colorbar(*args,ax=self.ax,pad=0.0,**kwargs)
	#     self.fig.colorbar(*args,ax=self.axes,pad=0.0,**kwargs)
	#     self.fig.colorbar(*args,ax=self.axes[:,0],pad=0.0,**kwargs)
	#     #self.fig.colorbar(*args,ax=self.ax,fraction=0.1,pad=0.0,**kwargs)
	
	# def figcolorbar(self,*args,**kwargs):
	#     self.fig.colorbar(*args,ax=self.axes,**kwargs)
	

	
	def tight_layout(self):
		self.fig.tight_layout()
		if (self.ymulti and self.xmulti):
			self.fig.subplots_adjust(wspace=0.0,hspace=0.0)
		elif self.ymulti and (self.xmulti==False):
			self.fig.subplots_adjust(wspace=0.0)
		elif self.xmulti and (self.ymulti==False):
			self.fig.subplots_adjust(hspace=0.0)
		for ax in self.axes.flat:
			plt.setp(ax.get_yticklabels()[-1],visible=False)
			plt.setp(ax.get_xticklabels()[-1],visible=False)
		#     for ax in self.axes.flat:
		#         ax.xaxis.set_major_locator(MaxNLocator(5,prune='upper'))
	
	
	
	def xoffset(self):
		ax=self.ax
		ax.set_xlabel('{0} {1}'.format(ax.get_xlabel(), ax.get_xaxis().get_offset_text().get_text()))
		ax.get_xaxis().get_offset_text().set_visible(False)
	
	
	def yoffset(self):
		ax=self.ax
		offset = ax.get_yaxis().get_offset_text()
		ax.set_ylabel('{0} {1}'.format(ax.get_ylabel(), offset.get_text()))
		offset.set_visible(False)
	
	def text(self,x,y,label, **kwargs):
		self.ax.text(x,y,label,transform=self.ax.transAxes,**kwargs)
	


class Plm1(object):

	def __init__(self,rows=1,cols=1,xsize=7.0,ysize=7.0,xmulti=False,ymulti=False,full=False,slabelx=0.89,slabely=0.1,slabel=True,order='left-right',slabclr='black'):
		self.slabels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		params = {'font.size':12,
				  'text.usetex':False,
				  'ytick.labelsize': 'medium',
				  'legend.fontsize': 'large',
				  'axes.linewidth': 1.0,
				  'xtick.labelsize': 'medium',
				  'font.family': 'sans-serif',
				  'axes.labelsize': 'medium'}
		mpl.rcParams.update(params)
	
	
		# if xsize>13.0:
		#     mpl.rcParams['font.size']=15
		#     mpl.rcParams['legend.fontsize']=15
		#mpl.rcParams['text.usetex'] = True
		if full:
			#mpl.rcParams['axes.titlesize']=8
			mpl.rcParams['font.size']=8
			mpl.rcParams['legend.fontsize']=8
		else: 
			mpl.rcParams['font.size']=15
			mpl.rcParams['legend.fontsize']=15
	
		self.rows=rows
		self.cols=cols
		self.figno=0
		self.panelno=0
		self.xmulti=xmulti
		self.ymulti=ymulti
		self.slabelx=slabelx
		self.slabely=slabely
		self.slabel=slabel
		self.order=order
		self.fig=plt.figure(figsize=(xsize,ysize))
		self.ax=None
		self.slabclr = slabclr
	
	def panels(self):
		return self.rows*self.cols
		
		
		
		
	def next(self,slabclr='black',weight='normal'):        
		self.slabclr = slabclr		
		self.weight = weight		
		self.figno+=1
		self.figno=(self.figno-1) % (self.rows*self.cols)+1
		if type(self.order) == list:
			self.panelno=self.order[self.figno-1]
		else:
			if self.order == 'top-down':
				r=(self.figno-1)/self.rows
				c=(self.figno-1)%self.rows
				self.panelno=(c*self.cols+r)+1
			else:
				self.panelno=self.figno
	
		plt.subplot(self.rows,self.cols,self.panelno)
		if (self.slabel) and (self.rows*self.cols > 1):
			plt.text(self.slabelx,self.slabely,'('+self.slabels[(self.panelno-1)%len(self.slabels)]+')',transform=plt.gca().transAxes,color=self.slabclr,weight=self.weight)
	
		if self.xmulti:
			if (self.panelno-1)/self.cols != (self.rows-1):
				plt.gca().set_xticklabels('')
				#ax.set_xticks([])
		if self.ymulti:
			if (self.panelno-1) % self.cols != 0:				
				plt.gca().set_yticklabels('')
				#ax.set_yticks([])
							
	
	def xlabel(self,s):
		if self.xmulti:
			if (self.panelno-1)/self.cols == (self.rows-1):
				plt.xlabel(s)		
				
		else:
			plt.xlabel(s)
		
	def ylabel(self,s):        
		if self.ymulti:
			if (self.panelno-1) % self.cols == 0:
				plt.ylabel(s)
		else:
			plt.ylabel(s)
		
	def colorbar(self,*args,**kwargs):
		plt.colorbar(fraction=0.1,pad=0.0,**kwargs)
		# if self.ymulti:
		#     if (self.panelno-1) % self.cols == (self.cols-1):
		#         plt.colorbar(fraction=0.1,pad=0.0,**kwargs)
		# else:
		#     plt.colorbar(fraction=0.1,pad=0.0,**kwargs)
	
	
	def tight_layout(self):
		plt.tight_layout()
		if (self.ymulti and self.xmulti):
			plt.subplots_adjust(wspace=0.0,hspace=0.0)
		elif self.ymulti and (self.xmulti==False):
			plt.subplots_adjust(wspace=0.0)
		elif self.xmulti and (self.ymulti==False):
			plt.subplots_adjust(hspace=0.0)
	
	def xoffset(self):
		ax=plt.gca()
		ax.set_xlabel('{0} {1}'.format(ax.get_xlabel(), ax.get_xaxis().get_offset_text().get_text()))
		ax.get_xaxis().get_offset_text().set_visible(False)
	
	
	def yoffset(self):
		ax=plt.gca()
		offset = ax.get_yaxis().get_offset_text()
		ax.set_ylabel('{0} {1}'.format(ax.get_ylabel(), offset.get_text()))
		offset.set_visible(False)
	
	def text(self,x,y,label, **kwargs):
		ax=plt.gca()
		ax.text(x,y,label,transform=ax.transAxes,**kwargs)
	
	
