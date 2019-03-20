#!/usr/bin/env python
"""Plot expansion history inferences.
"""

import argparse

import numpy as np
# matplotlib is imported inside main()
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import stats
#from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import astropy.constants as const
import astropy.cosmology


import gphist

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input',type = str, default = None,
        help = 'name of input file to read (extension .npz will be added)')
    parser.add_argument('--posterior', type = str, action='append', metavar = 'NAME',
        help = 'posteriors to plot (can be repeated, plot all if omitted)')
    parser.add_argument('--nlp', action = 'store_true',
        help = 'show plots of posterior -log(P) marginalized over hyperparameters')
    parser.add_argument('--full', action = 'store_true',
        help = 'show plots of DH/DH0,DA/DA0 evolution over the full redshift range')
    parser.add_argument('--zoom', action = 'store_true',
        help = 'show plots of DH,DA on a linear scale up to redshift zmax')
    parser.add_argument('--zmax', type = float, default = 3.0,
        help = 'maximum redshift to display on H(z)/(1+z) plot')
    parser.add_argument('--level', type = float, default = 0.9,
        help = 'confidence level to plot')
    parser.add_argument('--examples', action = 'store_true',
        help = 'include examples of random realizations in each plot')
    parser.add_argument('--output', type = str, default = None,
        help = 'base name for saving plots (no plots are saved if not set)')
    parser.add_argument('--show', action = 'store_true',
        help = 'show each plot (in addition to saving it if output is set)')
    parser.add_argument('--plot-format', type = str, default = 'png', metavar = 'FMT',
        help = 'format for saving plots (png,pdf,...)')
    parser.add_argument('--cosmo', type = float, default=0,
        help = 'use LCDM or CPL for plotting true cosmology')
    args = parser.parse_args()

    # Do we have any inputs to read?
    if args.input is None:
        print('Missing required input arg.')
        return -1

    # Do we have anything to plot?
    if not args.output and not args.show:
        print('No output requested.')
    #if args.examples:
    #    print('Option --examples not implemented yet.')
#        return -1

    # Initialize matplotlib.
    import matplotlib as mpl
    if not args.show:
        # Use the default backend, which does not require X11 on unix systems.
        mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import rc

    rc('font',**{'family':'serif','serif':['Times New Roman']})
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams.update({'font.size': 20})
    # Load the input file.
    loaded = np.load(args.input + '.npz')
    DH_hist = loaded['DH_hist']
    DA_hist = loaded['DA_hist']
    DH0 = loaded['DH0']
    DA0 = loaded['DA0']
    zvalues = loaded['zvalues']
    lna=-np.log(1 + zvalues[::-1])
    fixed_options = loaded['fixed_options']
    bin_range = loaded['bin_range']
    hyper_range = loaded['hyper_range']
    posterior_names = loaded['posterior_names']
    #if args.examples:
    #    DH_realizations = loaded['DH_realizations']
    #    DA_realizations = loaded['DA_realizations']

    #w0wa = astropy.cosmology.Flatw0waCDM(H0=73.,Om0=0.25,w0=-0.9,wa=-0.75)
    if args.cosmo==0:
        cosmo = astropy.cosmology.FlatLambdaCDM(H0=69.,Om0=0.3)
    elif args.cosmo==1:
        cosmo = astropy.cosmology.Flatw0waCDM(H0=69.,Om0=0.3,w0=-0.9,wa=-0.75)
    else:
        cosmo = astropy.cosmology.Flatw0waCDM(H0=69.,Om0=0.3,w0=-1.14,wa=0.35)





    # The -log(P) array is only present if this file was written by combine.py



    # Initialize the posterior permutations.
    npost = len(posterior_names)
    perms = gphist.analysis.get_permutations(npost)
    # Initialize the hyperparameter grid.
    n_samples,n_h,n_sigma = fixed_options
    h_min,h_max,sigma_min,sigma_max = hyper_range
    hyper_grid = gphist.process.HyperParameterLogGrid(
        n_h,h_min,h_max,n_sigma,sigma_min,sigma_max)

    # Loop over posterior permutations.
    for iperm,perm in enumerate(perms):

        name = '-'.join(posterior_names[perms[iperm]]) or 'Prior'
        if args.posterior and name not in args.posterior:
            continue
        print('%d : %s' % (iperm,name))



        #if num_plot_rows > 0:

            # Calculate the confidence bands of DH/DH0 and DA/DA0.
        DH_ratio_limits = gphist.analysis.calculate_confidence_limits(
            DH_hist[iperm],[args.level],bin_range)
        DA_ratio_limits = gphist.analysis.calculate_confidence_limits(
            DA_hist[iperm],[args.level],bin_range)
        DH_ratio_limits_1sig = gphist.analysis.calculate_confidence_limits(
            DH_hist[iperm],[0.68],bin_range)
        DA_ratio_limits_1sig = gphist.analysis.calculate_confidence_limits(
            DA_hist[iperm],[0.68],bin_range)

        # Convert to limits on DH, DA, with DA limits extended to z=0.
        DH_limits = DH_ratio_limits*DH0
        DA_limits = np.empty_like(DH_limits)
        DA_limits[:,1:] = DA_ratio_limits*DA0[1:]
        DA_limits[:,0] = 0.

        H_true = cosmo.H(zvalues).value
        DH_true = (const.c/cosmo.H(zvalues)).to('Mpc').value
        DA_true = cosmo.comoving_distance(zvalues).value

        # Find first z index beyond zmax.
        iend = 1+np.argmax(zvalues > args.zmax)

        fig = plt.figure(name,figsize=(8,8))
        fig.subplots_adjust(left=0.14,bottom=0.08,right=0.995,
            top=0.995,wspace=0.18,hspace=0.0)
        fig.set_facecolor('white')
        irow = 0

    # Plot evolution of DH/DH0, DA/DA0 over full redshift range.
    #if args.full:

        plt.subplot(2,1,2*irow+1)
        #plt.xscale('log')
        #plt.grid(True)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.xlim([0,np.max(zvalues)])
        plt.plot(zvalues,DH_true/DH0,'r')
        plt.fill_between(zvalues,DH_ratio_limits[0],DH_ratio_limits[-1],
            color=np.array((55,126,184))/255.,alpha=0.3)
        plt.fill_between(zvalues,DH_ratio_limits_1sig[0],DH_ratio_limits_1sig[-1],
            color=np.array((55,126,184))/255.,alpha=0.4)
        #plt.plot(zvalues,DH_ratio_limits[0],color=np.array((55,126,184))/255.,alpha=0.4)
        plt.plot(zvalues,DH_ratio_limits[1],color=np.array((55,126,184))/255.,linewidth=4.0)
        #plt.plot(zvalues,DH_ratio_limits[2],color=np.array((55,126,184))/255.,alpha=0.4)
        #plt.plot(zvalues,DH_ratio_limits_1sig[0],color=np.array((55,126,184))/255.,alpha=0.4)
        plt.plot(zvalues,DH_ratio_limits_1sig[1],color=np.array((55,126,184))/255.,linewidth=4.0)
        #plt.plot(zvalues,DH_ratio_limits_1sig[2],color=np.array((55,126,184))/255.,alpha=0.4)
        #if args.examples:
        #    for i in range(DH_realizations.shape[1]):
        #        plt.plot(1+zvalues,DH_realizations[0,i,:]/DH0)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$D_H(z)/D_H^0(z)$')

        plt.subplot(2,1,2*irow+2)
        #plt.xscale('log')
        #plt.grid(True)
        plt.xlim([0,np.max(zvalues)])
        plt.plot(zvalues,DA_true/DA0,'r')
        plt.fill_between(zvalues[1:],DA_ratio_limits[0],DA_ratio_limits[-1],
            color=np.array((55,126,184))/255.,alpha=0.3)
        plt.fill_between(zvalues[1:],DA_ratio_limits_1sig[0],DA_ratio_limits_1sig[-1],
            color=np.array((55,126,184))/255.,alpha=0.4)
        #plt.plot(zvalues[1:],DA_ratio_limits[0],'b:')
        plt.plot(zvalues[1:],DA_ratio_limits[1],color=np.array((55,126,184))/255.,linewidth=4.0)
        #plt.plot(zvalues[1:],DA_ratio_limits[2],'b:')
        #if args.examples:
        #    for i in range(DA_realizations.shape[1]):
        #        plt.plot(1+zvalues,DA_realizations[0,i,:]/DA0)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$D_A(z)/D_A^0(z)$')


        if args.output:
            plt.savefig(args.output + name + '.' + args.plot_format)

        plt.close()

        #print(DH_hist[iperm].shape)
        num_bins = 1001
        #print(bin_range)
        min_value,max_value = bin_range # Raises ValueError unless exactly 2 values to unpack.
        bin_edges = np.linspace(min_value,max_value,num_bins+1,endpoint=True)
        bins_smooth = np.linspace(min_value,max_value,101,endpoint=True)
        #kde_dh_hist = stats.gaussian_kde(DH_hist[iperm,0,0:-1])
        #yhat = savgol_filter(DH_hist[iperm,0,1:-1], 501, 2)
        #plt.plot(bin_edges[1:-1],yhat)
        #def gauss_func(x,a,b,c):
            #return a*np.exp(-(x-b)**2/c)
        #popt, pcov = curve_fit(gauss_func, bin_edges[1:-1], DH_hist[iperm,0,1:-1], p0= (1000,1,0.01), bounds=(0,[np.inf,2,0.5]))
        #print(popt)
        DH_arr = DH_hist[iperm,0,1:-1].reshape(100,10)
        sigma = 0.5*(DH_ratio_limits_1sig[0,0]-DH_ratio_limits_1sig[-1,0])
        mu = DH_ratio_limits_1sig[1,0]
        bin2 = np.linspace(mu-3*sigma,mu+3*sigma,101,endpoint=True)
        DH_bin = DH_arr.sum(axis=1)/10
        bin_arr = bin_edges[1:-1].reshape(100,10)
        bin_bin = bin_arr.sum(axis=1)/10

        fig = plt.figure(name,figsize=(8,6))
        fig.subplots_adjust(left=0.1,bottom=0.14,right=0.99,
            top=0.99)
        plt.ylabel('Relative Probability')
        plt.xlabel('$H_0$ [km/s/Mpc]')
        #rint(bin_bin)
        #plt.plot(bin_bin,DH_bin)
        #plt.plot(bin_edges[1:-1],np.exp(-(bin_edges[1:-1]-mu)**2/(2*sigma**2)))
        plt.plot(const.c.to('km/s').value/(DH0[0]*bin2),np.exp(-(bin2-mu)**2/(2*sigma**2)),linewidth=4.0)
        #plt.plot(bin_edges[1:-1],DH_hist[iperm,0,1:-1])
        #plt.plot(bin_edges[1:-1], gauss_func(bin_edges[1:-1], *popt))
        plt.savefig(args.output + name + '_H0post.' + args.plot_format)
        plt.clf()





if __name__ == '__main__':
    main()
