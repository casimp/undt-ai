# Data Description

The synthetic and experimental measurements have all been downsampled
to 300 points across a 12us window at a sampling rate of 25MHz. Note that the chosen window allows for a minimum of 2 reflections. Beyond 12us the quality of the synthetic signal (signal-to-noise ratio) is very poor.

### Synthetic UNDT measurements

The synthetic data is stored in .npz files, which behave like a python dictionary. Each file contains the time (t) and signal (signal) traces alongside the input parameters (params), describe the overall shape of the surface profile, which is 19mm x 19mm compared a transducer size that is, on average, 6mm x 6mmm. The surface profile directly under the transducer has been 'measured'. The mean thcickness under the transducer (D), minimum and maximum thickness (Dmin, Dmax) and roughness (ra, rms, rv, rp, rt) are all duly recorded under the transducer. The first 5796 runs (out of 12416) have a 6mm x 6mm aperture, the final 6620 runs have varying apertures.

- **synth_flat.npz**: synthetic measurements made on flat plates for various thicknesses (6 - 15mm)
- **synth_flat_vel.npz**: velocity corrected version of the synth_flat (corrected wrt. experimental measurements).
- **synth_flat_fft.npz**: frequency corrected version of the synth_flat (corrected wrt. experimental measurements).
- **synth_flat_fft_vel.npz**: frequency and velocity corrected version of the synth_flat (corrected wrt. experimental measurements).

- **synth_data.npz**: synthetic measurements made across an array of thicknesses, roughnesses, using multiple aperture sizes.
- **synth_data_vel.npz**: velocity corrected version of the synth_data (corrected wrt. experimental measurements).
- **synth_data_fft.npz**: frequency corrected version of the synth_data (corrected wrt. experimental measurements).
- **synth_data_fft_vel.npz**: frequency and velocity corrected version of the synth_data (corrected wrt. experimental measurements).

- **synth_data_params.pkl**: parameters for each run from synth_data. Includes the run number/id.

Note that the velocity and frequency conversions are carried out in the following notebooks: Synthetic_vs_Experimental-Freq_Correction.ipynb, Synthetic_vs_Experimental-Velocity_Freq_Correction.ipynb.

### Synthetic Input Signal

This synthetic input signal was also used for the experimental measurements but a further frequency transformation was required.

- **synth_inputSignal.txt** - raw input signal used in FEA simulations


### Experimental UNDT measurements

Experimental measurements were made on a section of rough pipe and a small number of coupons that were 
machined according to the profiles defined for the FE modles. The pipe measurements are supported by laser 
profilometry measurements across the pipe inner diameter (thereby defining the surface profile).

- **exp_coupons.npz**: experimental measurements on the coupons.
- **exp_pipe_surfA.npz**: experimental measurements across profile A from rough pipe.
- **exp_pipe_surfB.npz**: experimental measurements across profile B from rough pipe.
- **exp_pipe_random.npz**: experimental measurements at random points across rough pipe; not measured using profilometry.
- **exp_flat.npz**: experimental measurements made on a flat plate for a thickness of 10mm.


- **profile_surfA.pkl**: profile from which the exp_pipe_surfA measurements were made.
- **profile_surfB.pkl**: profile from which the exp_pipe_surfB measurements were made.