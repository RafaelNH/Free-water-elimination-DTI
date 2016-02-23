#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2016, Rafael Neto Henriques
#
# Contributors : Rafael Neto Henriques (rafaelnh21@gmail.com)
# -------------------------------------------------------------------------
# References:
#
# Hoy, A.R., Koay, C.G., Kecskemeti, S.R., Alexander, A.L., 2014
# Optimization of a free water elimination two-compartmental model
# for diffusion tensor imaging. NeuroImage 103, 323-333.
# doi: 10.1016/j.neuroimage.2014.09.053
# -------------------------------------------------------------------------


import numpy as np

from dipy.reconst.dti import (decompose_tensor, from_lower_triangular,
                              lower_triangular)

import scipy.optimize as opt

# -------------------------------------------------------------------------
# Weigthed linear least squares fit procedure
# -------------------------------------------------------------------------

def wls_fit_tensor(design_matrix, data, Diso=3e-3, piterations=3):
    r""" Weighted least squares (WLS) solution of the free water elimination
    DTI model.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    piterations : inter, optional
        Number of iterations used to refine the precision of f. Default is set
        to 3 corresponding to a precision of 0.01.

    Returns
    -------
    All parameters estimated from the free water tensor model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the
           first, second and third coordinates of the eigenvector
        3) The volume fraction of the free water compartment
        4) The estimate of the non diffusion-weighted signal S0
    """
    tol = 1e-6

    # preparing data and initializing parametres
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    fw_params = np.empty((len(data_flat), 14))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping WLS solution on all data voxels
    for vox in range(len(data_flat)):
        fw_params[vox] = _wls_iter(design_matrix, inv_design, data_flat[vox],
                                    min_diffusivity, Diso=Diso,
                                    piterations=piterations)

    # Reshape data according to the input data shape
    fw_params = fw_params.reshape((data.shape[:-1]) + (14,))

    return fw_params


def _wls_iter(design_matrix, inv_design, sig, min_diffusivity, Diso=3e-3,
              piterations=3):
    """ Helper function used by wls_fit_tensor - Applies WLS fit of the
    water free elimination model to single voxel signals.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    inv_design : array (g, 7)
        Inverse of the design matrix.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    piterations : inter, optional
        Number of iterations used to refine the precision of f. Default is set
        to 3 corresponding to a precision of 0.01.

    Returns
    -------
    All parameters estimated from the free water tensor model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the
           first, second and third coordinates of the eigenvector
        3) The volume fraction of the free water compartment
        4) The estimate of the non diffusion-weighted signal S0
    """
    W = design_matrix

    # DTI ordinary linear least square solution
    log_s = np.log(sig)

    # Define weights
    S2 = np.diag(sig**2)

    # DTI weighted linear least square solution
    WTS2 = np.dot(W.T, S2)
    inv_WT_S2_W = np.linalg.pinv(np.dot(WTS2, W))
    invWTS2W_WTS2 = np.dot(inv_WT_S2_W, WTS2)
    params = np.dot(invWTS2W_WTS2, log_s)

    # General free-water signal contribution
    fwsig = np.exp(np.dot(design_matrix, 
                          np.array([Diso, 0, Diso, 0, 0, Diso, 0])))

    df = 1  # initialize precision
    flow = 0  # lower f evaluated
    fhig = 1  # higher f evaluated
    ns = 9  # initial number of samples per iteration
    nvol = len(sig)
    for p in range(piterations):
        df = df * 0.1
        fs = np.linspace(flow+df, fhig-df, num=ns)  # sampling f
        # repeat fw contribution for all the samples
        SFW = np.array([fwsig,]*ns)
        FS, SI = np.meshgrid(fs, sig)
        # Free-water adjusted signal
        S0 = np.exp(-params[6])
        SA = SI - FS*S0*SFW.T
        # SA < 0 means that the signal components from the free water
        # component is larger than the total fiber. This cases are present
        # for inapropriate large volume fractions (given the current S0
        # value estimated). To avoid the log of negative values:
        SA[SA <= 0] = 0.0001  # same min signal assumed in dti.py
        y = np.log(SA / (1-FS))

        # Estimate tissue's tensor from inv(A.T*S2*A)*A.T*S2*y
        WTS2 = np.dot(W.T, S2)
        inv_WT_S2_W = np.linalg.pinv(np.dot(WTS2, W))
        invWTS2W_WTS2 = np.dot(inv_WT_S2_W, WTS2)
        all_new_params = np.dot(invWTS2W_WTS2, y)

        # compute F2
        S0r = np.exp(-np.array([all_new_params[6],]*nvol))
        SIpred = (1-FS)*np.exp(np.dot(W, all_new_params)) + FS*S0r*SFW.T
        F2 = np.sum(np.square(SI - SIpred), axis=0)

        # Select params for lower F2
        Mind = np.argmin(F2)
        params = all_new_params[:, Mind]

        # Updated f
        f = fs[Mind]
        # refining precision
        flow = f - df
        fhig = f + df
        ns = 19

    S0 = np.exp(-params[6])

    evals, evecs = decompose_tensor(from_lower_triangular(params),
                                    min_diffusivity=min_diffusivity)
    fw_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2], 
                                np.array([f]), np.array([S0])), axis=0)
    return fw_params

# -------------------------------------------------------------------------
# non-linear least squares fit procedure
# -------------------------------------------------------------------------

def nlls_fit_tensor(design_matrix, data, fw_params=None, Diso=3e-3):
    """
    Fit the water elimination tensor model using the non-linear least-squares.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.

    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    fw_params: ([X, Y, Z, ...], 14), optional
           A first model parameters guess (3 eigenvalues, 3 coordinates
           of 3 eigenvalues, the volume fraction of the free water
           compartment, and the estimate of the non diffusion-weighted signal
           S0). If the initial fw_paramters are not given, function will use
           the WLS free water elimination algorithm to estimate the parameters
           first guess.
           Default: None

    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.

    Returns
    -------
    All parameters estimated from the free water tensor model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the
           first, second and third coordinates of the eigenvector
        3) The volume fraction of the free water compartment
        4) The estimate of the non diffusion-weighted signal S0
    """
    # Flatten for the iteration over voxels:
    flat_data = data.reshape((-1, data.shape[-1]))

    # Use the WLS method parameters as the starting point if fw_params is None:
    if fw_params==None:
        fw_params = wls_fit_tensor(design_matrix, flat_data,  Diso=Diso)

    for vox in range(flat_data.shape[0]):
        if np.all(flat_data[vox] == 0):
            raise ValueError("The data in this voxel contains only zeros")

        params = fw_params[vox]

        # converting evals and evecs to diffusion tensor elements
        evals = params[:3]
        evecs = params[3:12].reshape((3, 3))
        dt = lower_triangular(np.dot(np.dot(evecs, np.diag(evals)), evecs.T))
        f = params[12]
        s0 = params[13]
        start_params = np.concatenate((dt, [-np.log(s0), f]), axis=0)

        this_tensor, status = opt.leastsq(_nlls_err_func, start_params[:8],
                                          args=(design_matrix,
                                                flat_data[vox],
                                                Diso))

        # The parameters are the evals and the evecs:
        try:
            evals, evecs = decompose_tensor(
                               from_lower_triangular(this_tensor[:6]))
            fw_params[vox, :3] = evals
            fw_params[vox, 3:12] = evecs.ravel()
            fw_params[vox, 12] = this_tensor[7]
            fw_params[vox, 13] = np.exp(-this_tensor[6])
        # If leastsq failed to converge and produced nans, we'll resort to the
        # WLS solution in this voxel:
        except np.linalg.LinAlgError:
            evals, evecs = decompose_tensor(
                              from_lower_triangular(start_params[:6]))
            fw_params[vox, :3] = evals
            fw_params[vox, 3:] = evecs.ravel()
            fw_params[vox, 12] = start_params[7]
            fw_params[vox, 13] = np.exp(-start_params[6])

    fw_params.shape = data.shape[:-1] + (14,)
    return fw_params


def _nlls_err_func(tensor_elements, design_matrix, data, Diso=3e-3):
    """ Error function for the non-linear least-squares fit of the tensor water
    elimination model.

    Parameters
    ----------
    tensor_elements : array (8, )
        The six independent elements of the diffusion tensor followed by
        -log(S0) and the volume fraction f of the water elimination compartment

    design_matrix : array
        The design matrix

    data : array
        The voxel signal in all gradient directions
    """
    f = tensor_elements[7]
    # This is the predicted signal given the params:
    y = (1-f) * np.exp(np.dot(design_matrix, tensor_elements[:7])) + \
        f * np.exp(np.dot(design_matrix,
                          np.array([Diso, 0, Diso, 0, 0, Diso, 
                                    tensor_elements[6]])))

    # Compute the residuals
    residuals = data - y

    return residuals
