---
Title: "Re: Optimization of a free water elimination two-compartment model for diffusion tensor imaging."
Author:
  - name: Rafael Neto Henriques
    affiliation: 1
  - name: Ariel Rokem
    affiliation: 2
  - name: Eleftherios Garyfallidis
    affiliation: 3
  - name: Samuel St-Jean
    affiliation: 4
  - name: Eric Thomas Peterson
    affiliation: 5
  - name: Marta Morgado Correia
    affiliation: 1
Address:
  - code:    1
    address: MRC Cognition and Brain Sciences Unit, Cambridge, Cambridgeshire, UK
  - code:    2
    address: The University of Washington eScience Institute, Seattle, WA, USA
  - code: 3
    address: Indiana University School of Informatics and Computing, Indiana, IA, USA
  - code: 4
    address: University Medical Center Utrecht, Utrecht, NL
  - code: 5
    address: Biosceinces, SRI International, Menlo Park, CA, USA
Contact:
  - rafaelnh21@gmail.com
Editor:
  - Name Surname
Reviewer:
  - Name Surname
  - Name Surname
Publication:
  received:  Sep,  1, 2015
  accepted:  Sep, 1, 2015
  published: Sep, 1, 2015
  volume:    "**1**"
  issue:     "**1**"
  date:      Sep 2015
Repository:
  article:   "http://github.com/rescience/rescience-submission/article"
  code:      "http://github.com/rescience/rescience-submission/code"
  data:      
  notebook:  
Reproduction:
  - "Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L. (2014).
Optimization of a free water elimination two-compartment model for diffusion
tensor imaging. NeuroImage 103, 323-333. doi: 10.1016/j.neuroimage.2014.09.053
"
Bibliography:
  bibliography.bib

---

# Introduction

Diffusion-weighted Magnetic Resonance Imaging (DW-MRI) is a
biomedical imaging technique that allows the non-invasive acquisition of in vivo data
from which tissue microstructure can be inferred. Diffusion tensor imaging (DTI), one of the most
commonly used DW-MRI techniques in the brain, models diffusion anisotropy of tissues using a
second-order tensor known as the diffusion tensor (DT) [@Basser1994-zd], [@Basser1994-hg].
DTI-based measures such as fractional anisotropy (FA) and mean diffusivity (MD)
are normally used to assess properties of brain microstructure. For example, FA is thought to be an indicator of different microstructural properties:
packing density of axons, and the density of myelin in nerve fibers [@Beaulieu2002-tl],
but also indicates white matter coherence -- the alignment of axons within a measurement voxel.
However, because a measurement region can contain multiple
types of tissue, these measures are not always specific to one particular type, an effect called partial voluming.
For example, diffusion anisotropy in regions near the cerebral ventricle and parenchyma can be
underestimated by partial volume effects of cerebral spinal fluid (CSF). To
remove the influence of the freely diffusing CSF which is not typically of interest, the DTI model
can be expanded to separately take into account the contributions tissue and CSF by representing
the tissue compartments with an anisotropic diffusion tensor and the CSF compartment as an isotropic
free water diffusion coefficient of $3.0 \times 10^{-3}  mm^{2}.s^{-1}$. Recently, two procedures were
proposed by Hoy and colleagues to fit this two compartment model to
diffusion-weighted data acquired with two or more diffusion gradient-weightings [@Hoy2014-lk,].
Although these procedures have been shown to provide diffusion based measures stable
to different degrees of free water contamination, the authors noted that their
original algorithms were "implemented by a group member with no formal programming
training and without optimization for speed" [@Hoy2014-lk,]. In this work, we provide
the first open-source reference implementation of the free water contamination DTI
model. All implementations are made in Python based on the descriptions provided
in Hoy et al.'s original article. For speed optimization, all necessary standard
DT processing steps used previously optimized functions freely available with the software
package Diffusion Imaging in Python (Dipy, http://nipy.org/dipy/,  [@Garyfallidis2014-zo])
and the optimization algorithms provided by the open-source software for mathematics,
science, and engineering Scipy (http://scipy.org/).

# Methods

## Implementation of fitting algorithms

Since no source implementation was previously provided by Hoy and colleagues,
our implementation relies on the equations provided in the original article.

**Weighted-Linear Least Square (WLS).** Two errors were found
in the equations describing the first proposed algorithm. Firstly, the free-water adjusted
diffusion-weighted signal formula (original article's Methods subsection
"FWE-DTI") should be written as:

$$y_{ik} = \ln\left\{ \frac{s_i - s_0 f_k\exp(-bD_{iso})}{(1-f_k)} \right \}$$ {#eq:1}

Secondly, according to the general linear least squares solution [@Jones2010-pg],
the weighted linear least squares solution to the free-water elimination model's
matrix solution should be given as:

$$\gamma = (W^TS^2W)^{-1}W^{T}S^{2}y$$ {#eq:2}

Thirdly, to ensure that the WLS method converges to the local minima,
the second and third iterations are used to refine the precision and therefore,
the water contamination volume fraction was resampled with steps sizes of 0.1 and 0.01
instead of the step sizes of 0.05 and 0.005 suggested by Hoy and colleagues.

Moreover, since the WLS objective function is sensitive to the squared error
of the model weights, when evaluating which ($f$, $D_tissue$) pair is associated
with smaller residuals the NLS objective function is used instead:

$$F_{NLS} = \frac{1}{2} \sum_{m}^{i=1} \left
 [s_{i} - S_{0} f\exp(-\sum_{j=2}^{4}W_{ij}D_{iso})
- (1-f)\exp(-\sum_{j=1}^{7}W_{ij}\gamma_{j})\right ]^{2}$$ {#eq:3}

Similarly to the original article [@Hoy2014-lk,], the WLS procedure is only used here
to obtain the intial guess for the free water elimination parameters, which were then
used to initialize the non-linear convergence procedure (see below).

**Non-Linear Least Square Solution (NLS)**. As suggested by Hoy and colleagues
[@Hoy2014-lk,], the initial guess for the non-linear convergence
procedure was set to the values estimated from the WLS approach. To improve the computation speed,
instead of using the modified Newton's algorithm proposed in the original article,
the non-linear convergence was done using Scipy's wrapped modified Levenberg-Marquardt algorithm
(function `scipy.optimize.leastsq` of Scipy http://scipy.org/). To constrain the
model parameters to within a plysically plausible range [0-1], the free water volume fraction $f$ was
converted to $f_t = \arcsin (2f-1) + \pi / 2$. To compare the robustness of the
techniques with and without this constraint, the free water volume fraction transformation was implemented
as an optional feature that can be controlled through user-provided arguments. In addition to
the `scipy.optimize.leastsq` function, a more recently implemented version of Scipy's optimization function
`scipy.optimize.least_square` (available as of Scipy's version 0.17) was also tested.
The latter directly solves the non-linear problem with predefined
constraints in a similar fashion to what is done in the original article, however
our experiments showed that this procedure does not overcome the performance of
`scipy.optimize.leastsq` in terms of accuracy, and requires more computing time
(see supplementary_notebook_1.ipynb for more details). To speed up the
performance of the non-linear optimization procedure, the jacobian of the free water elimination DTI model
was analytically derived and incorporated in the non-linear procedure (for details
of the jacobian derivation see supplementary_notebook_2.ipynb). As an expansion of
the work done by Hoy and colleagues, we also allow users to use the Cholesky
decomposition of the diffusion tensor to ensure that this is a positive definite tensor
[@Koay2006-zo]. Due to increased mathematical complexity, the Cholesky decomposition
is not used by default and it is not compatible with the analytical jacobian
derivation.

**Removing problematic estimates** For cases where the ground truth free water volume fraction is 1 (i.e. voxels
containing only free water), the tissue's diffusion tensor component can erroneously fit
the free water diffusion signal rather than placing the free water signal in the free water compartment,
and therefore incorrectly estimate the water volume fraction close to 0 rather than 1.
To remove these problematic cases, for all voxels with
standard DTI mean diffusivity values larger than $2.7 \times 10^{-3} mm^{2}.s^{-1}$, the free
water volume fraction is set to one while all other diffusion tensor
parameters are set to zero. This mean diffusivity threshold was arbitray adjusted to 90%
of the theoretical free water diffusion value, however this can be adjusted by
changing the optional input 'mdreg' in both WLS and NLS free water elimination
procedures.

**Implementation Dependencies**. In addition to the dependency on Scipy, both
free water elimination fitting procedures require modules from Dipy [@Garyfallidis2014-zo],
since these contain all necessary standard diffusion tensor fitting functions.
Although the core algorithms for the free water elimination model are implemented here separately from Dipy,
a version of these will be incorporated as a sub-module of Dipy's model
reconstruction module (https://github.com/nipy/dipy/pull/835). In addition, the
implemented procedures also requires the python pakage NumPy (http://www.numpy.org/),
which is also a dependency of both Scipy and Dipy.

## Simulations
In their original study, Hoy and colleagues simulated a measurement along 32 diffusion
direction with diffusion weighting b-values of 500 and 1500 $s.mm^{-2}$ and with six b-value=0 images.
These simulations correspond to the results reported in Fig. 5 of the original article.
We conducted Monte Carlo simulations using the multi-tensor simulation
module available in Dipy and using identical simulated acquisition parameters.
As in the original article, fitting procedures are tested for voxels with five different FA values
and with constant diffusion trace of $2.4 \times 10^{-3} mm^{2}.s^{-1}$.
The eigenvalues used for the five FA levels are reported in Table @tbl:table.

Table: Eigenvalues values used for the simulations {#tbl:table}

FA            0                      0.11                   0.22                   0.3                    0.71
------------ ---------------------- ---------------------- ---------------------- ---------------------- ----------------------
$\lambda_1$  $8.00 \times 10^{-4}$  $9.00 \times 10^{-4}$  $1.00 \times 10^{-3}$  $1.08 \times 10^{-3}$  $1.60 \times 10^{-3}$
$\lambda_2$  $8.00 \times 10^{-4}$  $7.63 \times 10^{-4}$  $7.25 \times 10^{-4}$  $6.95 \times 10^{-4}$  $5.00 \times 10^{-4}$
$\lambda_3$  $8.00 \times 10^{-4}$  $7.38 \times 10^{-4}$  $6.75 \times 10^{-4}$  $6.25 \times 10^{-4}$  $3.00 \times 10^{-4}$

For each FA value, eleven different degrees of free water contamination were
evaluated (f values equally spaced from 0 to 1). To assess the robustness of the
procedure, Rician noise with signal-to-noise ratio (SNR) of 40 relative to the b-value=0 images was
used. For each FA and f-value pair, simulations were performed for 120
different diffusion tensor orientation. Simulations for each diffusion tensor
orientation were repeated 100 times making a total of 12000 simulation
iterations for each FA and f-value pair.

## In vivo data

Similarly to the original article, the procedures are also tested using in vivo human brain data [@valabregue2015], that can be automatically
downloaded by Dipy's functions. The original dataset consisted of 74 volumes of images acquired for a
b-value of $0 s.mm^{-2}$ and 578 volumes diffusion weighted images acquired along 16 diffusion gradient directions
for b-values of 200 and 400 $s.mm^{-2}$ and along 182 diffusion gradient directions for b-values
of 1000, 2000 and 3000 $s.mm^{-2}$. In this study, only the data for b-values up to $2000 $s.mm^{-2}$
are used to decrease the impact of non-Gaussian diffusion effects which are not
taken into account by the free water elimination model. We also processed the data with the standard DTI tensor model
(as implemented in Dipy) in order to compare the results with the free water elimination model.

# Results

The results from the Monte Carlo simulations are shown in Figure @fig:simulations. Similarly to what is reported
in the original article, FA values estimated using the free water elimination model match the tissue's ground truth
values for free water volume fractions $f$ ranging around 0 to 0.7 (top panel of
Figure @fig:simulations). However, FA values seem to be overestimated for higher volume fractions. This bias is more
prominent for lower FA values in which overestimations are visible from lower free water volume
fractions. The lower panels of Figure @fig:simulations suggest that the free water elimination model produce
accurate free water volume fraction for the full range of volume fraction ground truth values. All the features observed
here are consistent with Fig. 5 of the original article.

![Fractional Anisotropy (FA) and free water volume fraction ($f$) estimates obtained from the Monte Carlo simulations
using the free water elimination fitting procedures. The top panel shows the FA median and intra-quartile range
for the five different FA ground truth levels and plotted as a function of the ground truth water volume fraction.
The bottom panels show the estimated volume fraction $f$ median and intra-quartile range as a function of its ground truth values
(right and left panels correspond to the higher and lower FA values, respectively). This figure reproduces
Fig. 7 of the original article.](fwdti_simulations.png){#fig:simulations}

In vivo tensor statistics obtained from the free water elimination and standard DTI models
are shown in Figure @fig:invivo. Complete processing of all these measure took less than 1 hour
in an average Desktop and Laptop PC (~2GHz processor speed), while the reported processing time
by Hoy et al. was around 20 hours. The free water elimination model seems to produce higher values
of FA in general and lower values of MD relative to the metrics obtained from the standard DTI model.
These differences in FA and MD estimates are expected due to the suppression
of the isotropic diffusion of free water. However, unexpectedly
high amplitudes of FA are observed in the periventricular gray mater. As mentioned in the original article,
this FA overestimation is related to the inflated values of FA in voxels with high $f$ values and
can be mitigated by excluding voxels with high free water volume
fraction estimates (see supplementary_notebook_3.ipynb).

![In vivo diffusion measures obtained from the free water DTI and standard
   DTI. The values of FA for the free water DTI model, the standard DTI model and
   their difference are shown in the top panels (A-C),
   while respective MD values are shown in the bottom panels (D-F). In addition
   the free water volume fraction estimated from the free water DTI model is shown in
   panel G.](In_vivo_free_water_DTI_and_standard_DTI_measures.png){#fig:invivo}


# Conclusion

Despite the changes done to reduce the algorithm's execution time, the
implemented procedures to solve the free water elimination DTI model have comparable performance
in terms of accuracy to the original methods described by Hoy and colleagues [@Hoy2014-lk].
Based on similar Monte Carlo simulations with the same SNR used in the original article,
our results confirmed that the free water elimination DTI model is able to remove confounding effects
of fast diffusion for typical FA values of brain white matter. Similarly to
what was reported by Hoy and colleagues, the proposed procedures seem to generate
biased values of FA for free water volume fractions near 1. Nevertheless,
our results confirm that these problematic cases correspond to regions that are not typically
of interest in neuroimaging analysis (voxels associated with cerebral ventricles)
and might be removed by excluding voxels with measured volume fractions above a reasonable
threshold such as 0.7.

# Author Contributions

Conceptualization: RNH, AR, MMC.
Data Curation: RNH, AR, EG, SSTJ.
Formal Analysis: RNH. 
Funding Acquisition: RNH, AR.
Investigation: RNH.
Methodology: RNH, AR, EG.
Project Administration: RNH, MMC, AR, EG.
Resources: RNH, MMC, AR.
Software: RNH, AR, ETP, EF, SSTJ.
Supervision: MMC, AR.
Validation: AR, SSTJ, EG.
Visualization: RNH.
Writing - Original Draft Preparation: RNH.
Writing - Review & Editing: AR, MMC.


# Acknowledgments

Rafael Neto Henriques was funded by Fundação para a Ciência e Tecnologia FCT/MCE (PIDDAC) under grant SFRH/BD/80114/2012.

Ariel Rokem was funded through a grant from the Gordon \& Betty Moore Foundation and the Alfred P. Sloan Foundation to the University of Washington eScience Institute.

Thanks to Romain Valabregue, CENIR, Paris for providing the data used here.


# References
