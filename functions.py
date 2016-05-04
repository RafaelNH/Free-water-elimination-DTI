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

def _wls_iter(design_matrix, sig, min_diffusivity, min_signal, Diso=3e-3,
              piterations=3, S0=None):
    """ Helper function used by wls_fit_tensor - Applies WLS fit of the
    water free elimination model to single voxel signals.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: minimal signal in the data provided to `fit`.
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
    fw_params : ndarray (13,)
        Array containing in the last dimention the free water model parameters
        in the following order:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
               first, second and third coordinates of the eigenvector
            3) The volume fraction of the free water compartment
    S0 : float
        Final estimate of the non diffusion-weighted signal S0.
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
    if S0 is not None:
        for p in range(piterations):
            df = df * 0.1
            fs = np.linspace(flow+df, fhig-df, num=ns)  # sampling f
            SFW = np.array([fwsig, ]*ns)  # repeat contributions for all values
            FS, SI = np.meshgrid(fs, sig)
            SA = SI - FS*S0*SFW.T
            # SA < 0 means that the signal components from the free water
            # component is larger than the total fiber. This cases are present
            # for inapropriate large volume fractions (given the current S0
            # value estimated). To overcome this issue negative SA are replaced
            # by data's min positive signal. 
            SA[SA <= 0] = min_signal
            y = np.log(SA / (1-FS))
            all_new_params = np.dot(invWTS2W_WTS2, y)

            # Select params for lower F2
            SIpred = (1-FS)*np.exp(np.dot(W, all_new_params)) + FS*S0*SFW.T
            F2 = np.sum(np.square(SI - SIpred), axis=0)
            Mind = np.argmin(F2)
            params = all_new_params[:, Mind]
            f = fs[Mind]  # Updated f
            flow = f - df  # refining precision
            fhig = f + df
            ns = 19
    else:
        for p in range(piterations):
            df = df * 0.1
            fs = np.linspace(flow+df, fhig-df, num=ns)  # sampling f
            SFW = np.array([fwsig, ]*ns)  # repeat contributions for all values
            FS, SI = np.meshgrid(fs, sig)
            S0 = np.exp(-params[6])  # S0 is now taken as a model parameter
            SA = SI - FS*S0*SFW.T
            SA[SA <= 0] = min_signal  # Overcaming issue of negative SA
            y = np.log(SA / (1-FS))
            all_new_params = np.dot(invWTS2W_WTS2, y)

            # Select params for lower F2
            S0r = np.exp(-np.array([all_new_params[6], ]*nvol))
            SIpred = (1-FS)*np.exp(np.dot(W, all_new_params)) + FS*S0r*SFW.T
            F2 = np.sum(np.square(SI - SIpred), axis=0)
            Mind = np.argmin(F2)
            params = all_new_params[:, Mind]
            f = fs[Mind]  # Updated f
            flow = f - df  # refining precision
            fhig = f + df
            ns = 19
        S0 = np.exp(-params[6])

    evals, evecs = decompose_tensor(from_lower_triangular(params),
                                    min_diffusivity=min_diffusivity)
    fw_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                np.array([f]), np.array([S0])), axis=0)

    return fw_params, S0


def wls_fit_tensor(design_matrix, data, S0=None, Diso=3e-3, piterations=3,
                   mdreg=2.7e-3):
    r""" Computes weighted least squares (WLS) fit to calculate self-diffusion
    tensor using a linear regression model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    S0 : array ([X, Y, Z]), optional
        Non diffusion weighted signal (i.e. signal for b-value=0). If not
        given, S0 will be taken as a model parameter (which is likely to
        decrease methods robustness).
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    piterations : inter, optional
        Number of iterations used to refine the precision of f. Default is set
        to 3 corresponding to a precision of 0.01.
    mdreg : float, optimal
        DTI's mean diffusivity regularization threshold. If standard DTI
        diffusion tensor's mean diffusivity is almost near the free water
        diffusion value, the diffusion signal is assumed to be only free water
        diffusion (i.e. volume fraction will be set to 1 and tissue's diffusion
        parameters are set to zero). Default md_reg is 2.7e-3 $mm^{2}.s^{-1}$
        (corresponding to 90% of the free water diffusion value).        

    Returns
    -------
    fw_params : ndarray (x, y, z, 13)
        Matrix containing in the last dimention the free water model parameters
        in the following order:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
               first, second and third coordinates of the eigenvector
            3) The volume fraction of the free water compartment
    S0 : ndarray (x, y, z, 13)
        Final estimate of the non diffusion-weighted signal S0.

    References
    ----------
    .. [1] Chung, SW., Lu, Y., Henry, R.G., 2006. Comparison of bootstrap
       approaches for estimation of uncertainties of DTI parameters.
       NeuroImage 33, 531-541.
    """
    tol = 1e-6

    # preparing data and initializing parameters
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    fw_params = np.zeros((len(data_flat), 14))
    S0f = np.zeros(len(data_flat))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()

    # Computing WLS DTI solution for MD regularization
    dti_params = dti_wls_fit(design_matrix, data_flat)
    md = mean_diffusivity(dti_params[..., :3])
    cond = md > mdreg  # removal condition
    fw_params_p = fw_params[~cond, :]
    data_flat_p = data_flat[~cond, ]
    S0_p = np.zeros(len(data_flat_p))

    # looping WLS solution on all data voxels
    if S0 is None:
        for vox in range(len(data_flat_p)):
            fw_params_p[vox], S0_p[vox] = _wls_iter(design_matrix,
                                                    data_flat_p[vox],
                                                    min_diffusivity,
                                                    min_signal=tol,
                                                    Diso=Diso,
                                                    piterations=piterations)
    else:
        S0i = S0.copy()
        S0i = S0.ravel()
        S0i = S0i[~cond]
        for vox in range(len(data_flat_p)):
            fw_params_p[vox], S0_p[vox] = _wls_iter(design_matrix,
                                                    data_flat_p[vox],
                                                    min_diffusivity,
                                                    min_signal=tol,
                                                    Diso=Diso,
                                                    piterations=piterations,
                                                    S0=S0i[vox])

    # Reshape data according to the input data shape
    fw_params[~cond, :] = fw_params_p
    fw_params[cond, 12] = 1  # Only free water
    fw_params = fw_params.reshape((data.shape[:-1]) + (14,))
    S0f[~cond] = S0_p
    S0f[cond] = np.mean(data[cond, :] / 
                        np.exp(np.dot(design_matrix[..., :6],
                                      np.array([Diso, 0, Diso, 0, 0, Diso]))),
                        -1)  # Only free water
    S0f = S0f.reshape(data.shape[:-1])
    return fw_params, S0f


# -------------------------------------------------------------------------
# non-linear least squares fit procedure
# -------------------------------------------------------------------------


def _nls_err_func(tensor_elements, design_matrix, data, Diso=3e-3,
                  cholesky=False, f_transform=False):
    """ Error function for the non-linear least-squares fit of the tensor water
    elimination model.

    Parameters
    ----------
    tensor_elements : array (8, )
        The six independent elements of the diffusion tensor followed by
        -log(S0) and the volume fraction f of the water elimination
        compartment. Note that if cholesky is set to true, tensor elements are
        assumed to be written as Cholesky's decomposition elements. If
        f_transform is true, volume fraction f has to be converted to
        ft = arcsin(2*f - 1) + pi/2
    design_matrix : array
        The design matrix
    data : array
        The voxel signal in all gradient directions
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    cholesky : bool, optional
        If true, the diffusion tensor elements were decomposed using cholesky
        decomposition. See fwdti.nls_fit_tensor
        Default: False
    f_transform : bool, optional
        If true, the water volume fraction was converted to
        ft = arcsin(2*f - 1) + pi/2, insuring f estimates between 0 and 1.
        See fwdti.nls_fit_tensor
        Default: True
    """
    tensor = np.copy(tensor_elements)
    if cholesky:
        tensor[:6] = cholesky_to_lower_triangular(tensor[:6])

    if f_transform:
        f = 0.5 * (1 + np.sin(tensor[7] - np.pi/2))
    else:
        f = tensor[7]

    # This is the predicted signal given the params:
    y = (1-f) * np.exp(np.dot(design_matrix, tensor[:7])) + \
        f * np.exp(np.dot(design_matrix,
                          np.array([Diso, 0, Diso, 0, 0, Diso, tensor[6]])))

    # Compute the residuals
    residuals = data - y

    return residuals



def _nls_jacobian_func(tensor_elements, design_matrix, data, Diso=3e-3,
                       cholesky=False, f_transform=False):
    """The Jacobian is the first derivative of the least squares error
    function.

    Parameters
    ----------
    tensor_elements : array (8, )
        The six independent elements of the diffusion tensor followed by
        -log(S0) and the volume fraction f of the water elimination
        compartment. Note that if f_transform is true, volume fraction f is
        converted to ft = arcsin(2*f - 1) + pi/2
    design_matrix : array
        The design matrix
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    f_transform : bool, optional
        If true, the water volume fraction was converted to
        ft = arcsin(2*f - 1) + pi/2, insuring f estimates between 0 and 1.
        See fwdti.nls_fit_tensor
        Default: True
    """
    tensor = np.copy(tensor_elements)
    if f_transform:
        f = 0.5 * (1 + np.sin(tensor[7] - np.pi/2))
    else:
        f = tensor[7]

    t = np.exp(np.dot(design_matrix, tensor[:7]))
    s = np.exp(np.dot(design_matrix,
                      np.array([Diso, 0, Diso, 0, 0, Diso, tensor[6]])))
    T = (f-1.0) * t[:, None] * design_matrix
    S = np.zeros(design_matrix.shape)
    S[:, 6] = f * s

    if f_transform:
        df = (t-s) * (0.5*np.cos(tensor[7]-np.pi/2))
    else:
        df = (t-s)
    return np.concatenate((T - S, df[:, None]), axis=1)


def nls_fit_tensor(design_matrix, data, fw_params=None, S0=None, Diso=3e-3,
                   cholesky=False, f_transform=True, jac=False, mdreg=2.7e-3):
    """
    Fit the water elimination tensor model using the non-linear least-squares.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    fw_params : ndarray ([X, Y, Z, ...], 14), optional
        A first model parameters guess (3 eigenvalues, 3 coordinates
        of 3 eigenvalues, and the volume fraction of the free water
        compartment). If the initial fw_paramters are not given, the function
        will use free water DTI WLS solution. Default: None
    S0 : ndarray ([X, Y, Z])
        A first guess of the non-diffusion signal S0. If S0 and fw_params are
        not given, an initial guess of the S0 will be extracted from the free
        water WLS solution. If only S0 is not given, the function will use the
        S0 estimate from standard DTI WLS. Default: None
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    cholesky : bool, optional
        If true it uses cholesky decomposition to insure that diffusion tensor
        is positive define.
        Default: False
    f_transform : bool, optional
        If true, the water volume fractions is converted during the convergence
        procedure to ft = arcsin(2*f - 1) + pi/2, insuring f estimates between
        0 and 1.
        Default: True
    jac : bool
        Use the Jacobian? Default: False
    mdreg : float, optimal
        DTI's mean diffusivity regularization threshold. If standard DTI
        diffusion tensor's mean diffusivity is almost near the free water
        diffusion value, the diffusion signal is assumed to be only free water
        diffusion (i.e. volume fraction will be set to 1 and tissue's diffusion
        parameters are set to zero). Default md_reg is 2.7e-3 $mm^{2}.s^{-1}$
        (corresponding to 90% of the free water diffusion value).

    Returns
    -------
    fw_params : ndarray (x, y, z, 13)
        Matrix containing in the dimention the free water model parameters in
        the following order:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
               first, second and third coordinates of the eigenvector
            3) The volume fraction of the free water compartment
    S0 : ndarray (x, y, z, 13)
        The models estimate of the non diffusion-weighted signal S0.
    """
    # Flatten for the iteration over voxels:
    flat_data = data.reshape((-1, data.shape[-1]))

    # Computing WLS DTI solution for MD regularization
    dti_params = dti_wls_fit(design_matrix, flat_data)
    md = mean_diffusivity(dti_params[..., :3])
    cond = md > mdreg  # removal condition
    flat_data_p = flat_data[~cond, :]

    # Initializing fw_params according to selected initial guess
    if np.any(fw_params) is None:
        if np.any(S0) is None:
            fw_paramsc, S0 = wls_fit_tensor(design_matrix, flat_data,
                                            Diso=Diso)
        else:
            fw_paramsc, S0f = wls_fit_tensor(design_matrix, flat_data, S0=S0,
                                           Diso=Diso)
    else:
        fw_paramsc = fw_params.copy()
        fw_paramsc = fw_paramsc.reshape((-1, fw_params.shape[-1]))

    # Initializing S0 according to selected initial guess
    if np.any(S0) is None:
        evals = dti_params[..., :3]
        evecs = dti_params[..., 3:12].reshape((len(flat_data),) + (3, 3))
        dti_lower_tri = lower_triangular(vec_val_vect(evecs, evals))
        S0f = np.mean(flat_data /
                     np.exp(np.dot(dti_lower_tri, design_matrix[..., :6].T)),
                     -1)
    else:
        S0f = S0.copy()
        S0f = S0f.ravel()

    fw_params_p = fw_paramsc[~cond, :]
    S0_p = S0f[~cond]

    for vox in range(flat_data_p.shape[0]):
        if np.all(flat_data_p[vox] == 0):
            raise ValueError("The data in this voxel contains only zeros")

        params = fw_params_p[vox]

        # converting evals and evecs to diffusion tensor elements
        evals = params[:3]
        evecs = params[3:12].reshape((3, 3))
        dt = lower_triangular(vec_val_vect(evecs, evals))
        s0 = S0_p[vox]

        # Cholesky decomposition if requested
        if cholesky:
            dt = lower_triangular_to_cholesky(dt)

        # f transformation if requested
        if f_transform:
            f = np.arcsin(2*params[12] - 1) + np.pi/2
        else:
            f = params[12]

        # Use the Levenberg-Marquardt algorithm wrapped in opt.leastsq
        start_params = np.concatenate((dt, [-np.log(s0), f]), axis=0)
        if jac:
            this_tensor, status = opt.leastsq(_nls_err_func, start_params[:8],
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    Diso,
                                                    cholesky,
                                                    f_transform),
                                              Dfun=_nls_jacobian_func)
        else:
            this_tensor, status = opt.leastsq(_nls_err_func, start_params[:8],
                                              args=(design_matrix,
                                                    flat_data[vox],
                                                    Diso,
                                                    cholesky,
                                                    f_transform))


        # Invert the cholesky decomposition if this was requested
        if cholesky:
            this_tensor[:6] = cholesky_to_lower_triangular(this_tensor[:6])

        # Invert f transformation if this was requested
        if f_transform:
            this_tensor[7] = 0.5 * (1 + np.sin(this_tensor[7] - np.pi/2))

        # The parameters are the evals and the evecs:
        fw_params_p[vox, 12] = this_tensor[7]
        S0_p[vox] = np.exp(-this_tensor[6])
        evals, evecs = decompose_tensor(from_lower_triangular(this_tensor[:6]))
        fw_params_p[vox, :3] = evals
        fw_params_p[vox, 3:12] = evecs.ravel()

    fw_paramsc[~cond, :] = fw_params_p
    fw_paramsc[cond, 12] = 1  # Only free water
    fw_paramsc = fw_paramsc.reshape((data.shape[:-1]) + (14,))
    S0f[~cond] = S0_p
    S0f[cond] = np.mean(data[cond, :] / 
                        np.exp(np.dot(design_matrix[..., :6],
                                      np.array([Diso, 0, Diso, 0, 0, Diso]))),
                        -1)  # Only free water
    S0f = S0f.reshape(data.shape[:-1])
    return fw_paramsc, S0f


def lower_triangular_to_cholesky(tensor_elements):
    """ Perfoms Cholesky decompostion of the diffusion tensor

    Parameters
    ----------
    tensor_elements : array (6,)
        Array containing the six elements of diffusion tensor's lower
        triangular.
    Returns
    -------
    cholesky_elements : array (6,)
        Array containing the six Cholesky's decomposition elements
        (R0, R1, R2, R3, R4, R5) [1]_.
    References
    ----------
    .. [1] Koay, C.G., Carew, J.D., Alexander, A.L., Basser, P.J.,
           Meyerand, M.E., 2006. Investigation of anomalous estimates of
           tensor-derived quantities in diffusion tensor imaging. Magnetic
           Resonance in Medicine, 55(4), 930-936. doi:10.1002/mrm.20832
    """
    R0 = np.sqrt(tensor_elements[0])
    R3 = tensor_elements[1] / R0
    R1 = np.sqrt(tensor_elements[2] - R3**2)
    R5 = tensor_elements[3] / R0
    R4 = (tensor_elements[4] - R3*R5) / R1
    R2 = np.sqrt(tensor_elements[5] - R4**2 - R5**2)

    return np.array([R0, R1, R2, R3, R4, R5])


def cholesky_to_lower_triangular(R):
    """ Convert Cholesky decompostion elements to the diffusion tensor elements

    Parameters
    ----------
    R : array (6,)
        Array containing the six Cholesky's decomposition elements
        (R0, R1, R2, R3, R4, R5) [1]_.

    Returns
    -------
    tensor_elements : array (6,)
        Array containing the six elements of diffusion tensor's lower
        triangular.

    References
    ----------
    .. [1] Koay, C.G., Carew, J.D., Alexander, A.L., Basser, P.J.,
           Meyerand, M.E., 2006. Investigation of anomalous estimates of
           tensor-derived quantities in diffusion tensor imaging. Magnetic
           Resonance in Medicine, 55(4), 930-936. doi:10.1002/mrm.20832
    """
    Dxx = R[0]**2
    Dxy = R[0]*R[3]
    Dyy = R[1]**2 + R[3]**2
    Dxz = R[0]*R[5]
    Dyz = R[1]*R[4] + R[3]*R[5]
    Dzz = R[2]**2 + R[4]**2 + R[5]**2
    return np.array([Dxx, Dxy, Dyy, Dxz, Dyz, Dzz])


# -------------------------------------------------------------------------
# Supplementary function
# -------------------------------------------------------------------------


def nls_fit_tensor_bounds(design_matrix, data, fw_params=None, Diso=3e-3,
                           bounds=None, jac=False, mdreg=2.7e-3):
    """
    Fit the water elimination tensor model using the non-linear least-squares.

    Parameters
    ----------
    design_matrix : array (g, 7)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    fw_params : ndarray ([X, Y, Z, ...], 14), optional
        A first model parameters guess (3 eigenvalues, 3 coordinates
        of 3 eigenvalues, and the volume fraction of the free water
        compartment). If the initial fw_paramters are not given, the function
        will use free water DTI WLS solution. Default: None
    S0 : ndarray ([X, Y, Z])
        A first guess of the non-diffusion signal S0. If S0 and fw_params are
        not given, an initial guess of the S0 will be extracted from the free
        water WLS solution. If only S0 is not given, the function will use the
        S0 estimate from standard DTI WLS. Default: None
    Diso : float, optional
        Value of the free water isotropic diffusion. Default is set to 3e-3
        $mm^{2}.s^{-1}$. Please ajust this value if you are assuming different
        units of diffusion.
    bounds : 2-tuple of arrays with 14 elements, optional
        Lower and upper bounds on fwdti model variables and the non-diffusion
        signal S0. Use np.inf with an appropriate sign to disable bounds on all
        or some variables. When bounds is set to None the following default
        variable bounds is used:
            ([0., -Diso, 0., -Diso, -Diso, 0., 0., np.exp(-10.)],
             [Diso, Diso, Diso, Diso, Diso, Diso, 1., np.exp(10.)])
    jac : bool
        Use the Jacobian? Default: False
    mdreg : float, optimal
        DTI's mean diffusivity regularization threshold. If standard DTI
        diffusion tensor's mean diffusivity is almost near the free water
        diffusion value, the diffusion signal is assumed to be only free water
        diffusion (i.e. volume fraction will be set to 1 and tissue's diffusion
        parameters are set to zero). Default md_reg is 2.7e-3 $mm^{2}.s^{-1}$
        (corresponding to 90% of the free water diffusion value).

    Returns
    -------
    fw_params : ndarray (x, y, z, 13)
        Matrix containing in the dimention the free water model parameters in
        the following order:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
               first, second and third coordinates of the eigenvector
            3) The volume fraction of the free water compartment
    S0 : ndarray (x, y, z, 13)
        The models estimate of the non diffusion-weighted signal S0.
    """
    # Flatten for the iteration over voxels:
    flat_data = data.reshape((-1, data.shape[-1]))

    # Computing WLS DTI solution for MD regularization
    dti_params = dti_wls_fit(design_matrix, flat_data)
    md = mean_diffusivity(dti_params[..., :3])
    cond = md > mdreg  # removal condition
    flat_data_p = flat_data[~cond, :]

    # Set bounds
    if bounds==None:
        bounds = ([0., -Diso, 0., -Diso, -Diso, 0., -10., 0],
                  [Diso, Diso, Diso, Diso, Diso, Diso, 10., 1])
    else:
        # In the helper subfunctions it was easier to have log(S0) first than 
        # the water volume. Therefore, we have to reorder the boundaries if
        # specified by the user
        S0low = np.log(bounds[0][7])
        S0hig = np.log(bounds[1][7])
        bounds[0][7] = bounds[0][6]
        bounds[1][7] = bounds[1][6]
        bounds[0][6] = S0low
        bounds[1][6] = S0hig   

    # Initializing fw_params according to selected initial guess
    if np.any(fw_params) is None:
        if np.any(S0) is None:
            fw_paramsc, S0 = wls_fit_tensor(design_matrix, flat_data,
                                            Diso=Diso)
        else:
            fw_paramsc, S0f = wls_fit_tensor(design_matrix, flat_data, S0=S0,
                                           Diso=Diso)
    else:
        fw_paramsc = fw_params.copy()
        fw_paramsc = fw_paramsc.reshape((-1, fw_params.shape[-1]))

    # Initializing S0 according to selected initial guess
    if np.any(S0) is None:
        evals = dti_params[..., :3]
        evecs = dti_params[..., 3:12].reshape((len(flat_data),) + (3, 3))
        dti_lower_tri = lower_triangular(vec_val_vect(evecs, evals))
        S0f = np.mean(flat_data /
                     np.exp(np.dot(dti_lower_tri, design_matrix[..., :6].T)),
                     -1)
    else:
        S0f = S0.copy()
        S0f = S0f.ravel()

    fw_params_p = fw_paramsc[~cond, :]
    S0_p = S0f[~cond]

    for vox in range(flat_data_p.shape[0]):
        if np.all(flat_data_p[vox] == 0):
            raise ValueError("The data in this voxel contains only zeros")

        params = fw_params_p[vox]

        # converting evals and evecs to diffusion tensor elements
        evals = params[:3]
        evecs = params[3:12].reshape((3, 3))
        dt = lower_triangular(vec_val_vect(evecs, evals))
        s0 = S0_p[vox]

        # Use the Levenberg-Marquardt algorithm wrapped in opt.leastsq
        start_params = np.concatenate((dt, [-np.log(s0), f]), axis=0)
        lb = np.array(bounds[0])
        ub = np.array(bounds[1])
        start_params[start_params<lb] = lb[start_params<lb]
        start_params[start_params>ub] = ub[start_params>ub]       
        if jac:
            out = opt.least_squares(_nls_err_func, start_params[:8],
                                    args=(design_matrix, flat_data[vox],
                                          Diso, False, False),
                                    jac=_nls_jacobian_func,
                                    bounds=bounds)
        else:
            out = opt.least_squares(_nls_err_func, start_params[:8],
                                    args=(design_matrix, flat_data[vox],
                                          Diso, False, False),
                                    bounds=bounds)
        this_tensor = out.x

        # The parameters are the evals and the evecs:
        fw_params_p[vox, 12] = this_tensor[7]
        S0_p[vox] = np.exp(-this_tensor[6])
        evals, evecs = decompose_tensor(from_lower_triangular(this_tensor[:6]))
        fw_params_p[vox, :3] = evals
        fw_params_p[vox, 3:12] = evecs.ravel()

    fw_paramsc[~cond, :] = fw_params_p
    fw_paramsc[cond, 12] = 1  # Only free water
    fw_paramsc = fw_paramsc.reshape((data.shape[:-1]) + (14,))
    S0f[~cond] = S0_p
    S0f[cond] = np.mean(data[cond, :] / 
                        np.exp(np.dot(design_matrix[..., :6],
                                      np.array([Diso, 0, Diso, 0, 0, Diso]))),
                        -1)  # Only free water
    S0f = S0f.reshape(data.shape[:-1])
    return fw_paramsc, S0f

