#!/usr/bin/env python
"""Infer the cosmological expansion history using a Gaussian process prior.
"""

import argparse
import math

import numpy as np

import gphist
import matplotlib.pyplot as plt

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type = int, default = 26102014,
        help = 'random seed to use for sampling the prior')
    parser.add_argument('--num-samples', type = int, default = 1000000,
        help = 'number of samples to generate')
    parser.add_argument('--num-evol-hist', type = int, default = 100,
        help = 'number of equally spaced evolution variable steps to use for histogramming')
    parser.add_argument('--max-array-size', type = float, default = 1.0,
        help = 'maximum array memory allocation size in gigabytes')
    parser.add_argument('--hyper-h', type = float, default = 0.01,
        help = 'vertical scale hyperparameter value to use')
    parser.add_argument('--hyper-sigma', type = float, default = 0.01,
        help = 'horizontal scale hyperparameter value to use')
    parser.add_argument('--hyper-index', type = int, default = None,
        help = 'index into hyperparameter marginalization grid to use (ignore if None)')
    parser.add_argument('--hyper-count', type = int, default = 1,
        help = 'number of consecutive marginalization grid indices to run')
    parser.add_argument('--hyper-num-h', type = int, default = 10,
        help = 'number of h grid points in marginalization grid')
    parser.add_argument('--hyper-h-min', type = float, default = 0.001,
        help = 'minimum value of hyperparameter h for marginalization grid')
    parser.add_argument('--hyper-h-max', type = float, default = 0.1,
        help = 'maximum value of hyperparameter h for marginalization grid')
    parser.add_argument('--hyper-num-sigma', type = int, default = 20,
        help = 'number of sigma grid points in marginalization grid')
    parser.add_argument('--hyper-sigma-min', type = float, default = 0.001,
        help = 'minimum value of hyperparameter sigma for marginalization grid')
    parser.add_argument('--hyper-sigma-max', type = float, default = 2.0,
        help = 'maximum value of hyperparameter sigma for marginalization grid')
    parser.add_argument('--omega-k', type = float, default =0.,
        help = 'curvature parameter')
    parser.add_argument('--cosmo', type = int, default =0,
        help = 'choose whether to use LCDM or w0wa or w0wa2 mock GW,SN DL')
    parser.add_argument('--dark-energy', action= 'store_true',
        help = 'calculate dark energy expansion history for each realization')
    parser.add_argument('--growth', action= 'store_true',
        help = 'calculate growth functions for each realization')
    parser.add_argument('--accel', action= 'store_true',
        help = 'calculate deceleration parameter for each realization')
    parser.add_argument('--num-bins', type = int, default = 1000,
        help = 'number of bins to use for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--min-ratio', type = float, default = 0.8,
        help = 'minimum ratio for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--max-ratio', type = float, default = 1.2,
        help = 'maximum ratio for histogramming DH/DH0 and DA/DA0')
    parser.add_argument('--num-save', type = int, default = 0,
        help = 'number of prior realizations to save for each combination of posteriors')
    parser.add_argument('--output', type = str, default = None,
        help = 'name of output file to write (the extension .npz will be added)')
    args = parser.parse_args()





    # Initialize the posteriors to use.
    posteriors = [
    ]

    if args.cosmo ==0:
        #GW_data = np.loadtxt('GW_DL_LCDM.txt')
        GW_data = np.loadtxt('GW_LCDM_600_test2.txt')
        #SN_data = np.loadtxt('sn_Pantheon_LCDM_91537.txt')
        #SN_forc = np.loadtxt('sn_WFIRST_LCDM_654987.txt')
        SN_data = np.loadtxt('sn_pan_lcdm_20.txt')
        SN_forc = np.loadtxt('sn_wfirst_lcdm_20.txt')
        print('LCDM')
    elif args.cosmo ==1:
        #GW_data = np.loadtxt('GW_w0wa_600.txt')
        #SN_data = np.loadtxt('sn_Pantheon_CPL_91537.txt')
        #SN_forc = np.loadtxt('sn_WFIRST_CPL_654987.txt')
        GW_data = np.loadtxt('GW_w0wa_600_99.txt')
        SN_data = np.loadtxt('sn_pan_cpl_92.txt')
        SN_forc = np.loadtxt('sn_wfirst_cpl_5.txt')
        print('w0wa')
    else:
        GW_data = np.loadtxt('GW_w0wa2_600_42.txt')
        SN_data = np.loadtxt('sn_pan_cpl2_27.txt')
        SN_forc = np.loadtxt('sn_wfirst_cpl2_2.txt')
        print('w0wa2')

    z_GW = GW_data[:,0]
    mean_GW = GW_data[:,1]/(1+z_GW)
    err_GW = GW_data[:,2]/(1+z_GW)

    z_SN = SN_data[:,0]
    #z_SN = np.round(z_SN,4)
    print(z_SN.shape)
    z_SN,i_SN = np.unique(z_SN,return_index=True)
    print(i_SN.shape)
    mean_SN = SN_data[i_SN,1]
    err_SN = SN_data[i_SN,2]

    z_forc = SN_forc[:,0]
    mean_forc = SN_forc[:,1]
    err_forc = SN_forc[:,2]


    GW_post = gphist.posterior.GWPosterior('GW',z_GW,mean_GW,err_GW)
    SN_post = gphist.posterior.SNPosterior('SN',z_SN,mean_SN,err_SN)
    SN2_post = gphist.posterior.SNPosterior('Wfirst',z_forc,mean_forc,err_forc)

    #I dont think these are doing anything -- maybe just initializing the empty array
    posterior_names = np.array([p.name for p in posteriors])
    posterior_redshifts = np.array([p.zpost for p in posteriors])

    posterior_redshifts = np.concatenate((posterior_redshifts,GW_post.zpost))
    posterior_redshifts = np.concatenate((posterior_redshifts,SN_post.zpost))
    posterior_redshifts = np.concatenate((posterior_redshifts,SN2_post.zpost))


    posterior_names = np.concatenate((posterior_names,np.array([GW_post.name])))
    posterior_names = np.concatenate((posterior_names,np.array([SN_post.name])))
    posterior_names = np.concatenate((posterior_names,np.array([SN2_post.name])))


    posteriors.append(GW_post)
    posteriors.append(SN_post)
    posteriors.append(SN2_post)


    # Initialize a grid of hyperparameters, if requested.
    if args.hyper_index is not None:
        hyper_grid = gphist.process.HyperParameterLogGrid(
            args.hyper_num_h,args.hyper_h_min,args.hyper_h_max,
            args.hyper_num_sigma,args.hyper_sigma_min,args.hyper_sigma_max)
    else:
        hyper_grid = None

    # Loop over hyperparameter values.
    for hyper_offset in range(args.hyper_count):

        if hyper_grid:
            hyper_index = args.hyper_index + hyper_offset
            h,sigma = hyper_grid.get_values(hyper_index)
        else:
            hyper_index = None
            h,sigma = args.hyper_h,args.hyper_sigma

        print('Using hyperparameters (h,sigma) = (%f,%f)' % (h,sigma))

        # Initialize the Gaussian process prior.
        prior = gphist.process.SquaredExponentialGaussianProcess(h,sigma)

        # Calculate the amount of oversampling required in the evolution variable to
        # sample the prior given this value of sigma.
        min_num_evol = math.ceil(2./sigma)
        num_evol,evol_oversampling,samples_per_cycle = gphist.evolution.initialize(
            min_num_evol,args.num_evol_hist,args.num_samples,args.max_array_size)

        print('Using %dx oversampling and %d cycles of %d samples/cycle.' % (
            evol_oversampling,math.ceil(1.*args.num_samples/samples_per_cycle),
            samples_per_cycle))

        # Initialize the evolution variable.
        evol = gphist.evolution.LogScale(num_evol,evol_oversampling,posterior_redshifts)
        #iprior = np.where(np.in1d(evol.zvalues,z_SN))[0]
        #iprior2 = np.where(np.in1d(evol.zvalues,z_GW))[0]
        #print(iprior.shape)
        #print(iprior2.shape)

        # Initialize the distance model.
        model = gphist.distance.HubbleDistanceModel(evol)
        DH0 = model.DH0
        DA0 = model.DC0 # assuming zero curvature


        # Initialize a reproducible random state for this offset. We use independent
        # states for sampling the prior and selecting random realizations so that these
        # are independently reproducible.
        sampling_random_state = np.random.RandomState([args.seed,hyper_offset])
        realization_random_state = np.random.RandomState([args.seed,hyper_offset])

        # Break the calculation into cycles to limit the memory consumption.
        combined_DH_hist,combined_DA_hist,combined_q_hist= None,None,None
        samples_remaining = args.num_samples
        while samples_remaining > 0:

            samples_per_cycle = min(samples_per_cycle,samples_remaining)
            samples_remaining -= samples_per_cycle

            # Generate samples from the prior.
            samples = prior.generate_samples(samples_per_cycle,evol.svalues,
                sampling_random_state)

            # Convert each sample into a corresponding tabulated DH(z).
            DH = model.get_DH(samples)
            del samples


            # Free the large sample array before allocating a large array for DC.


            # Calculate the corresponding comoving distance functions DC(z).
            DC = evol.get_DC(DH)

            # Calculate the corresponding comoving angular scale functions DA(z).
            DA = gphist.distance.convert_DC_to_DA(DH,DC,args.omega_k)
            mu = evol.get_mu2(DH,DA,evol.zvalues)
            #print(mu)
            #print(evol.zvalues.shape)
            #print('mu shape in infer')
            #print(mu.shape)

            # Calculate -logL for each combination of posterior and prior sample.
            posteriors_nlp = gphist.analysis.calculate_posteriors_nlp(
                evol.zvalues,DH,DA,mu,posteriors)#the emptys are for the aperp and apar needed for the r_s scaling case

            # Select some random realizations for each combination of posteriors.
            # For now, we just sample the first cycle but it might be better to sample
            # all cycles and then downsample.
            #if combined_DH_hist is None:
            #    DH_realizations,DA_realizations = gphist.analysis.select_random_realizations(
            #        DH,DA,posteriors_nlp,args.num_save,realization_random_state)

            # Downsample distance functions in preparation for histogramming.
            i_ds = evol.downsampled_indices
            z_ds,DH0_ds,DA0_ds = evol.zvalues[i_ds],DH0[i_ds],DA0[i_ds]
            DH_ds,DA_ds = DH[:,i_ds],DA[:,i_ds]

            if args.accel:
                print('calculating q')
                q = evol.get_accel(DH,evol.svalues)
                q0 = evol.get_accel(DH0[np.newaxis,:],evol.svalues)
                q_ds = q[:,i_ds]
                q0_ds = q0[:,i_ds]
                print('calculating q')
            else:
                q_ds = None
                q0_ds = None




            # Build histograms for each downsampled redshift slice and for
            # all permutations of posteriors.
            DH_hist,DA_hist,q_hist = gphist.analysis.calculate_histograms(
                DH_ds,DH0_ds,DA_ds,DA0_ds,q_ds,q0_ds,posteriors_nlp,
                args.num_bins,args.min_ratio,args.max_ratio)

            print('done histogramming')

            # Combine with the results of any previous cycles.
            if combined_DH_hist is None:
                combined_DH_hist = DH_hist
                combined_DA_hist = DA_hist
                if args.accel:
                    combined_q_hist = q_hist
            else:
                combined_DH_hist += DH_hist
                combined_DA_hist += DA_hist
                if args.accel:
                    combined_q_hist += q_hist

            print('Finished cycle with %5.2f%% samples remaining.' % (
                100.*samples_remaining/args.num_samples))

        # Save the combined results for these hyperparameters.
        if args.output:
            fixed_options = np.array([args.num_samples,
                args.hyper_num_h,args.hyper_num_sigma])
            variable_options = np.array([args.seed,hyper_index,hyper_offset])
            bin_range = np.array([args.min_ratio,args.max_ratio])
            hyper_range = np.array([args.hyper_h_min,args.hyper_h_max,
                args.hyper_sigma_min,args.hyper_sigma_max])
            output_name = '%s.%d.npz' % (args.output,hyper_offset)
            np.savez(output_name,
                zvalues=z_ds,
                DH_hist=combined_DH_hist,DA_hist=combined_DA_hist,q_hist=combined_q_hist,
                DH0=DH0_ds,DA0=DA0_ds,q0=q0_ds,
                fixed_options=fixed_options,variable_options=variable_options,
                bin_range=bin_range,hyper_range=hyper_range,
                #DH_realizations=DH_realizations,DA_realizations=DA_realizations,
                posterior_names=posterior_names)
            print('Wrote %s' % output_name)

if __name__ == '__main__':
    main()
