---
Title: "Re: Optimization of a free water elimination two-compartment model for diffusion tensor imaging."
Author:
  - name: Rafael Neto Henriques
    affiliation: 1
  - name: Ariel Rokem
    affiliation: 2
Address:
  - code:    1
    address: MRC, Cognition and Brain Sciences Unit, Cambridge, Cambridgeshire, UK
  - code:    2
    address: Affiliation Dept/Program/Center, Institution Name, City, State, Country
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
  article.bib

---

# Introduction

Diffusion-weighted Magnetic Resonance Imaging (DWI) is a non-invasive biomedical
imaging technique that allows us to infer properties of brain tissue
microstructures in vivo. Diffusion tensor imaging (DTI), one of the most
commonly used DWI techniques, models anisotropy diffusion of tissues using a
second-order tensor known as diffusion tensor (DT) [@Basser1994-zd, @Basser1994-hg].
DTI-based measures such as the fractional anisotropy (FA) are
normally used as an indicator of white matter coherence. However, these measures
are not always specific to one particular type of tissue. For example, diffusion
anisotropy in regions near the cerebral ventricle and parenchyma can be
underestimated by partial volume effects of the cerebral spinal fluid (CSF). To
remove the influence of this free water diffusion contamination, the DTI model
can be expanded to take into account two compartments representing the diffusion
contributions from the tissue and from the CSF. Recently, two procedures were
proposed by Hoy and colleagues to fit the free water elimination of
DWI data acquired with two or more diffusion gradient-weightings [@Hoy2014-lk,].
Although, these procedures are showed to provide diffusion based measures stable
to different free water contamination degrees, the authors mentioned that their
original algorithms were "implemented by a group member with no formal programming
training and without optimization for speed" [@Hoy2014-lk,]. In this work, we provide
the first open source reference implementation of the free water contamination DTI
model fitting procedures. All implementations are made in Python based on a
reviewed description of the Hoy and collegues original article. For speed optimization,
all necessary standard DT processing steps are using the previously optimized functions
of the software project Diffusion Imaging (Dipy, http://nipy.org/dipy/,  [@Garyfallidis2012-zp])
and the optimization algorithms provided by the open-source software for mathematics, science,
and engineering (Scipy, http://scipy.org/).

# Methods

## Procedures Implementation

Since no source implementation was previously provided by Hoy and
colleagues, all formulas of the two fitting procedures on the original paper were
carefully review.

**The Weighted-Linear Least Square Solution (WLS).** Two typos were found
on the formulas of the first proposed algorithm. First, the free-water adjusted
diffusion-weighted signal formula (original article's Methods subsection
"FWE-DTI") should be written as:

$$y_{ik} = \ln\left\{ \frac{s_i - s_0 f_k\exp(-bD_{iso})}{(1-f_k)} \right \}$$ {#eq:1}

Second, according to the general linear least squares solution [@Jones2010-pg],
the weighted linear least square solution of the free-water elimination model's
matrix solution should be given as:

$$\gamma = (W^TS^2W)^{-1}W^{T}S^{2}y$$ {#eq:2}

Moreover, to insure that the WLS method convergences to the local minima,
on the second and third iteration to refine the precision, the water
contamination volume fraction were resampled with steps sizes of 0.1 and 0.01
instead of the step sizes of 0.05 and 0.005 suggested by Hoy and Colleges.

**Non-Linear Least Square Solution (NLS)**. For the non-linear convergence
procedure, as suggested by Hoy and Colleges [@Hoy2014-lk,], the model parameters initial
guess were adjusted to the values estimated from the WLS approach. For computing
speed optiminzation, instead of using the modified Newton's method approach
proposed in the original article, the non-linear covergence was followed using
Scipy's wrapped modified levenberg-marquardt algorithm available (function
scipy.optimize.leastsq of Scipy http://scipy.org/). To constrain the
models parameters to plausibe range, the free water volume fraction $f$ was
converted to $f_t = \arcsin (2f-1) + \pi / 2$. To compare the robustness of the
techniques with and without this constrains, the free water volume fraction
transformation was implemented as an optional function feature. In addition to
the scipy.optimize.leastsq, the more recent Scipy's optimization function
scipy.optimize.least_square (available in Scipy's version 0.17) was also tested.
This allows solving the non-linear problems directly bounded with predefined
constrains similar to what is done on the original article, however for the
free water elimination model this did not show to overcome the robustness
and time speed of the procedure scipy.optimize.leastsq when the proposed f
transformation was used (see supplementary_notebook_1.ipynb for more details).
To speed the non-linear performance, the free water elimination DTI model jacobian
was analytically derived and incorporated to the non-linear procedure (for the details
of the jacobian derivation see supplementary_notebook_2.ipynb). As an expansion of
the work done by Hoy and colleagues, we also allow users to use of Cholesky
decomposition of the diffusion tensor to insure that this is a positive defined tensor
[@Koay2006-zo]. Due to the increase of the model's mathematical complexity, the
Cholesky decomposition is not used by default (see supplementary_notebook_1.ipynb
for more details).

**Removing problematic estimates**

For cases that the ground truth free water volume fraction is one (i.e. voxels
containing only free water), the tissue's diffusion tensor componet can erratically
overfit the free water diffusion signal and erratically induce estimates of
the water volume fraction near to one. To remove this problematic cases, for all voxels with
standard DTI's mean diffusivity values larger than 2.7 mm^{2}.s^{-1}, the free
water volume fraction is set to one while all tissue's diffusion tensor
parameters are set to zero. This mean diffusivity threshold was adjusted to 90%
of the theoretical free water diffusion value, however this can be adjusted by
changing the optional input 'mdreg' in both WLS and NLS free water elimination
procedures.

**Implementation Dependencies**. In addition to the Scipy's dependencies, both
free water elimination fitting procedures requires modules from the open source
software project Dipy  [@Garyfallidis2012-zp], since these contain all necessary
standard diffusion tensor processing functions. Although, the core algorithms of
the free water elimination procedures were implemented separately from Dipy,
in the near future, they will be incorporated as a Dipy's model reconstruction
module (https://github.com/nipy/dipy/pull/835). In addition, our functions also
requires the python pakage NumPy (http://www.numpy.org/).

### 2.2 Simulations
In this study, the Hoy and colleagues simulations for the methods optimal
acquisition parameters are reproduces (i.e. simulations along 32 diffusion
direction for b-values 500 and 1500 s.mm^{-2} and with six b-value=0 images).
The monte carlos simulations are performed using the multi-tensor simulation
module available in Dipy. As the original article, fitting procedures are
tested for voxels with 5 different FA values and with constant diffusion trace
of $2.4 \times 10^{-3} mm^{2}.s^{-1}$. The eigenvalues used for the 5 FA levels
are reported in @tbl:table.

Table: Eigenvalues values used for the simulations of the study {#tbl:table}

FA            0                      0.11                   0.22                   0.3                    0.71
------------ ---------------------- ---------------------- ---------------------- ---------------------- ----------------------
$\lambda_1$  $8.00 \times 10^{-4}$  $9.00 \times 10^{-4}$  $1.00 \times 10^{-3}$  $1.08 \times 10^{-3}$  $1.60 \times 10^{-3}$
$\lambda_2$  $8.00 \times 10^{-4}$  $7.63 \times 10^{-4}$  $7.25 \times 10^{-4}$  $6.95 \times 10^{-4}$  $5.00 \times 10^{-4}$
$\lambda_3$  $8.00 \times 10^{-4}$  $7.38 \times 10^{-4}$  $6.75 \times 10^{-4}$  $6.25 \times 10^{-4}$  $3.00 \times 10^{-4}$

For each FA value, eleven different degrees of free water contamination were
evaluated (f values equaly spaced from 0 to 1). To access the robustness of the
procedure, Rician noise with a SNR of 40 relative to the b-value = 0 images was
used. For each FA and f-value pair, simulations were performed for 120
different diffusion tensor orientation. Simulations for each diffusion tensor
orientation were repeated 100 times making a total of 12000 simulation
iterations for each FA and f-value pair.

## Real data testing

# Results

A reference to figure @fig:logo.

![Figure caption](rescience-logo.pdf) {#fig:logo}

# Conclusion

Despite the changes done to improve the algorithms speed performance, the
implemented procedures to solve the free water elimination DTI model show to
have identical performance to the original methods described by Hoy and
Colleagues [@Hoy2014-lk]. Based on similar monte-carlo simulations and
signal to noise rations used on the original article, ours results confirmed
that the free water elimination DTI model is able to remove confounding effects
of fast diffusion for typical FA values of the brain white matter. Similar to
what was reported by Hoy and Colleagues, the proposed procedures seem to produce
biased values of FA for free water volume fractions near to one. Nevertheless,
our results confirm that these problematic cases correspond to regions that are not
of interest in neuroimaging analysis (voxels associated with cerebral ventricles)
and might be removed by excluding voxels with high volume free water volume
fractions estimates.


# References
