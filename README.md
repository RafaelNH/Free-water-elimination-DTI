# Free-water-elimination-DTI
This repository contains all necessary functions to fit the free water
elimination DTI (fwdti).

The fit procedures were based on the work proposed by Hoy et al. (2014). For 
more details on the implementation are reported in article.md. For an example 
of how to run this procedure in a real data please give a look to the ipython
notebook run_realdata.ipynb. For a quantitative evaluation of the technique
using monte carlo simulation please see the notebook run_sim1.ipynb
(this evaluation reproduces the results of Hoy et al. (2014)). Finally
in the notebook supplementary_info.ipynb, some of the details of the free
water elimination model implementations are explored (for instance this 
notebook will show why the scipy.optimize.leastsq is used for the non-linear
fwdti fit instead of using the must recent sicpy's optimeze module
scipy.optimize.least_squares).

When publishing or disclose any result obtained through the use of the
this procedures please include the following reference:

Neto Henriques, R. (...)

# References

1) Neto Henriques, R. (...)

2) Hoy et al. 2014 Optimization of a free water elimination two-compartment
model for diffusion tensor imaging. Neuroimage 103:323-33.
doi: 10.1016/j.neuroimage.2014.09.053
