"""Expansion history posterior applied to distance functions.
"""

import math
from abc import ABCMeta,abstractmethod

import numpy as np
import numpy.linalg
import astropy.constants

class GaussianPdf(object):
	"""Represents a multi-dimensional Gaussian probability density function.

	Args:
		mean(ndarray): 1D array of length npar of mean values.
		covariance(ndarray): 2D symmetric positive definite covariance matrix
			of shape (npar,npar).

	Raises:
		ValueError: dimensions of mean and covariance are incompatible.
		LinAlgError: covariance matrix is not positive definite.
	"""
	def __init__(self,mean,err):
		# Check that the dimensions match or throw a ValueError.
		#dimensions_check = mean.dot(covariance.dot(mean))
		# Check that the covariance is postive definite or throw a LinAlgError.
        #posdef_check = numpy.linalg.cholesky(covariance[:,:,0])

		self.mean = mean
		self.icov = 1/err**2
		#self.norm = 0.5*mean.size*np.log(2*math.pi) + 0.5*np.log(np.linalg.det(covariance))
		#self.norm = 0.5*mean.size*np.log(2*math.pi) - 0.5*np.log(err**2).sum()
		#print(mean.size)
		self.norm = -0.5*mean.size
		# Calculate the constant offset of -log(prob) due to the normalization factors.

	def get_nlp(self,values):
		"""Calculates -log(prob) for the PDF evaluated at specified values.

		The calculation is automatically broadcast over multiple value vectors.

		Args:
			values(ndarray): Array of values where the PDF should be evaluated with
				shape (neval,ndim) where ndim is the dimensionality of the PDF and
				neval is the number of points where the PDF should be evaluated.
			more precisely, it has shape (nsamples,ntype,nzposterior)
			nsamples is the number of samples requested
			ntype is the number of types of posteriors, types being DH, DA or mu
			nzposterior is the number of redshifts in a given posterior

		Returns:
			float: Array of length neval -log(prob) values calculated at each input point.

		Raises:
			ValueError: Values can not be broadcast together with our mean vector.
		"""
		# The next line will throw a ValueError if values cannot be broadcast.
		#print('values shape')
		#print(values.shape)
		#print('mean shape')
		#print(self.mean.shape)
		residuals = values - self.mean
        #a[2] is ntype; the difference between these cases which dimension the covariance matrix is for
        # ALL OF THESE RESIDUALS SHOULD BE OF THE FORM (NSAMPLE,Ndata)
		chisq = np.einsum('...ij,j,...ij->...i',residuals,self.icov,residuals)
		#print(chisq)
		print(chisq.min())
		#print(self.norm)
		#if a[1]>1 and a[2]==1:   # should correspond to just SN data
			#chisq = np.einsum('...ijk,jl,...ilk->...i',residuals,self.icov,residuals)
		#elif a[2]>1 and a[1]>1:  #should correspond to just BOSS 2016 w/ BOSS2018 should never happen
			#chisq = np.einsum('...ijk,klj,...ijl->...i',residuals,self.icov,residuals)
		#else:
			#chisq = np.einsum('...ijk,kl,...ijl->...i',residuals,self.icov,residuals)
		#print(self.norm + 0.5*chisq)
		return self.norm + 0.5*chisq
		#return 0.5*chisq

class GaussianPdf1D(GaussianPdf):
	"""Represents a specialization of GaussianPdf to the 1D case.

	Args:
		central_value(float): Central value of the 1D PDF.
		sigma(float): RMS spread of the 1D PDF.
	"""
	def __init__(self,central_value,sigma):
		mean = np.array([central_value])
		covariance = np.array([[sigma**2]])
		GaussianPdf.__init__(self,mean,covariance)

	def get_nlp(self,values):
		"""Calculates -log(prob) for the PDF evaluated at specified values.

		Args:
			values(ndarray): Array of values where the PDF should be evaluated with
				length neval.

		Returns:
			float: Array of length neval -log(prob) values calculated at each input point.
		"""
		return GaussianPdf.get_nlp(self,values)

class GaussianPdf2D(GaussianPdf):
	"""Represents a specialization of GaussianPdf to the 2D case.

	Args:
		x1(float): Central value of the first parameter.
		x2(float): Central value of the second parameter.
		sigma1(float): RMS spread of the first parameter.
		sigma2(float): RMS spread of the second parameter.
		rho12(float): Correlation coefficient between the two parameters. Must be
			between -1 and +1.
	"""
	def __init__(self,x1,sigma1,x2,sigma2,rho12):
		mean = np.array([x1,x2])
		cov12 = sigma1*sigma2*rho12
		covariance = np.array([[sigma1**2,cov12],[cov12,sigma2**2]])
		#mean = mean[np.newaxis,:]#see the CMB posterior class
		GaussianPdf.__init__(self,mean,covariance)

class Posterior(object):
	"""Posterior constraint on DH,DA at a fixed redshift.

	This is an abstract base class and subclasses must implement the constraint method.

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift of posterior constraint.
	"""
	__metaclass__ = ABCMeta
	def __init__(self,name,zpost):
		self.name = name
		self.zpost = zpost

	@abstractmethod
	def constraint(self,DHz,DAz,muz):
		"""Evaluate the posterior constraint given values of DH(zpost) and DA(zpost).

		Args:
			DHz(ndarray): Array of DH(zpost) values.
			DAz(ndarray): Array of DA(zpost) values with the same shape as DHz.

		Returns:
			nlp(ndarray): Array of -log(prob) values with the same shape as DHz and DAz.
		"""
		pass

	def get_nlp(self,zprior,DH,DA,mu):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		The posterior is applied to c/H(z=0).

			zprior(ndarray): Redshifts where prior is sampled, in increasing order.
			DH(ndarray): Array of shape (nsamples,nz) of DH(z) values to use.
			DA(ndarray): Array of shape (nsamples,nz) of DA(z) values to use.

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.

		Raises:
			AssertionError: zpost is not in zprior.
		"""
		#iprior = np.argmax(zprior==self.zpost)
		iprior = np.where(np.in1d(zprior,self.zpost))[0] #for whatever reason np.where returns a tuple of an array so thats why there is the [0] after
		DHz = DH[:,iprior]
		DAz = DA[:,iprior]
		muz = mu[:,iprior]# these should be of the form (nsample,nz)
		return self.constraint(DHz,DAz,muz)



class GWPosterior(Posterior):
	"""Posterior constraint on DH(z).

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift of posterior constraint.
		DC(float): Central values of DC(z).
		cov_Hz(float): cov of H(z).
	"""
	def __init__(self,name,zpost,DCz,err_DC):
		#print('GW posterior shape')
		#print(DCz.shape)
		self.pdf = GaussianPdf(DCz,err_DC)
		Posterior.__init__(self,name,zpost)

	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		Args:
			DHz(ndarray): Array of DH(zpost) values to use (will be ignored).
			DAz(ndarray): Array of DA(zpost) values to use.

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		#print('DAz shape')
		#print(DAz.shape)
		return self.pdf.get_nlp(DAz)

class SNPosterior(Posterior):
	"""Posterior constraint on mu(z).

	Args:
		name(str): Name to associate with this posterior.
		zpost(float): Redshift of posterior constraint.
		mu(float): Central value of mu*(z): actually mu(z)-(M_1=19.05).
		mu_error(float): RMS error on mu(z).
	"""
	def __init__(self,name,zpost,mu,mu_error):
		#print('SN posterior shape')
		#print(mu.shape)
		self.pdf = GaussianPdf(mu,mu_error)
		Posterior.__init__(self,name,zpost)


	def constraint(self,DHz,DAz,muz):
		"""Calculate -log(prob) for the posterior applied to a set of expansion histories.

		Args:
			DHz(ndarray): Array of DH(zpost) values to use (will be ignored).
			DAz(ndarray): Array of DA(zpost) values to use (also ignored).
			muz(ndarray): Array of mu(zpost) 5log(DL)+25

		Returns:
			ndarray: Array of -log(prob) values calculated at each input value.
		"""
		#print('muz shape')
		#print(DAz.shape)
		#print(muz.shape)
		return self.pdf.get_nlp(muz)
