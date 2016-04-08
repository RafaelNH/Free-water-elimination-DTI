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


# References

1) Hoy et al. 2014 Optimization of a free water elimination two-compartment
model for diffusion tensor imaging. Neuroimage 103:323-33.
doi: 10.1016/j.neuroimage.2014.09.053
