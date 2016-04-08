# Re: Optimization of a free water elimination two-compartment model for 
diffusion tensor imaging.

## Introduction

Diffusion-weighted Magnetic Resonance Imaging (DWI) is a non-invasive
biomedical imaging technique that allows us to infer properties of brain tissue
microstructures in vivo. Diffusion tensor imaging (DTI), one of the most
commonly used DWI techniques, models anisotropy diffusion of tissues using a
second-order tensor known as diffusion tensor (DT). DTI-based measures such as
the fractional anisotropy (FA) are normally used as an indicator of white
matter coherence. However, these measures are not always specific to one
particular type of tissue. For example, diffusion anisotropy in regions near
the cerebral ventricle and parenchyma can be underestimated by partial volume
effects of the cerebral spinal fluid (CSF). To remove the influence of this
free water diffusion contamination, the DTI model can be expanded to take into
account two compartments representing the diffusion contributions from the
tissue and from the CSF. Recently, two procedures were proposed by Hoy and
colleagues to fit the free water elimination of DWI data acquired with two or
more diffusion gradient-weightings [1]. Although the authors mentioned that
their algorithm were implemented "by a group member with no formal programming
training and without optimization for speed", their evaluation tests showed
that their procedures are able to provide diffusion based measures stable
to different free water contamination degree. In this work, the two algorithms
described by Hoy and colleagues are reviewed, optimized and tested on
monte-carlo simulations based also in the evaluations done by Hoy and
colleagues [1]. Our goal is to provide the first reference implementation of
the procedures to fit the free water contamination model optimized for both
speed and robustness.

## 2. Methods

### 2.1. Procedures Implementation
The two procedures to fit the free water elimination DTI model were implemented
in python. Since no source implementation was previously provided by Hoy and
colleagues, all formulas of the original paper were carefully review.

**The Weighted-Linear Least Square Solution (WLLS).** Two typos were found
on the formulas of this algorithm. First, the free-water adjusted
diffusion-weighted signal formula (original article's Methods subsection
‘FWE-DTI’) should be written as:

$$y_{ik} = \ln\left\{ \frac{s_i - s_0 f_k\exp(-bD_{iso})}{(1-f_k)} \right \}$$

Second, according to the general linear least squares solution [2], the
weighted linear least square solution of the free-water elimination model’s
matrix solution should be given as:

$$\gamma = (W^TS^2W)^{-1}W^{T}S^{2}y$$

Moreover, to insure that the WLLS method convergences to the local minima,
on the second and third iteration to refine the precision, the water
contamination volume fraction were resampled with steps sizes of ± .1 and ± .01
instead of the step sizes of ± .05 and ± .005 suggested by Hoy and Colleges.

**Non-Linear Least Square Solution (NLLS)**. For the non-linear convergence
procedure the parameters initial guest were adjusted to the values estimated
from the WLLS approach as suggested by Hoy and Colleges [1].

**Implemtation Dependencies**. Both fitting procedures requires also modules
from the open source software project Diffusion Imaging (Dipy,
http://nipy.org/dipy/) [3]. Although, for this study, the core of each
procedure was implemented in separate functions for the models parameter
estimation, the DT derived measures (diffusion eigenvalues, and diffusion
fractions anisotropy) are processed using the already implemented Dipy’s
standard DTI modules.

### 2.2 Simulations
The monte carlos simulations are performed using the multi-tensor simulation
model of the open source software project Diffusion Imaging in python
(Dipy, http://nipy.org/dipy/).

### 2.3 Real data testing

## 3. Results

## 4. Conclusion

Despite the changes done to improve the algorithms speed performance, the
implemented procedures to solve the free water elimination DTI model show to
have identical performance to the original methods described by Hoy and
Colleagues [1]. Ours results confirmed that the free water elimination DTI
model can be used to remove confounding effects of fast diffusion on the brain,
particulary for images voxels associated with high fractional anisotropy
values. Regarding to the problematic cases of voxels with high free water
volume fraction, we confirmed that this could be easly identified based on the
free water volume fraction estimates provided by the implemented model and
eventually removed.

# References

[1] Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L., 2014.
Optimization of a free water elimination two-compartment model for diffusion
tensor imaging. NeuroImage 103, 323-333. doi: 10.1016/j.neuroimage.2014.09.053

[2] Jones, D.K., Cercignani, M., 2010. Twenty-five pitfalls in the analysis
of diffusion MRI data. NMR in Biomedicine 23(7), 803-820.
doi: 10.1002/nbm.1543.

[3] Garyfallidis, E., Brett, M., Amirbekian, B., Rokem, A., van der Walt, S.,
Descoteaux, M., Nimmo-Smith, I., and Dipy Contributors, 2014. Dipy, a library
for the analysis of diffusion MRI data. Frontiers in Neuroinformatics 8 (8).
doi: 10.3389/fninf.2014.00008.