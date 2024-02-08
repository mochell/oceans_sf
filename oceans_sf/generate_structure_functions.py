import numpy as np

from .bin_data import bin_data
from .calculate_advection import calculate_advection
from .calculate_separation_distances import calculate_separation_distances
from .calculate_structure_function import (
    calculate_structure_function,
)
from .shift_array1d import shift_array1d

import concurrent.futures

def generate_structure_functions(  # noqa: C901, D417
    u,
    v,
    x,
    y,
    skip_velocity_sf=False,
    scalar=None,
    cal_shear =False,
    traditional_order=0,
    dx=None,
    dy=None,
    boundary="Periodic",
    even="True",
    grid_type="uniform",
    nbins=10,
    parallel = False,
    max_workers = 4
):
    """
    Full method for generating structure functions for 2D data, either advective or
    traditional structure functions. Supports velocity-based and scalar-based structure
    functions. Defaults to calculating the velocity-based advective structure functions
    for the x (zonal) and y (meridional) directions.

    Parameters
    ----------
        u: ndarray
            2D array of u velocity components.
        v: ndarray
            2D array of v velocity components.
        x: ndarray
            1D array of x-coordinates.
        y: ndarray
            1D array of y-coordinates.
        skip_velocity_sf: bool, optional
            Flag used to skip calculating the velocity-based structure function if
            the user only wants to calculate the scalar-based structure function.
            Defaults to False.
        scalar: ndarray, optional
            2D array of scalar values. Defaults to None.
        cal_shear: bool, optional
            Flag used to calculate the shear structure function. Defaults to False.
        traditional_order: int, optional
            Order for calculating traditional non-advective structure functions.
            If 0, no traditional structure functions are calculated. Defaults to 0.
        dx: float, optional
            Grid spacing in the x-direction. Defaults to None.
        dy: float, optional
            Grid spacing in the y-direction. Defaults to None.
        boundary: str, optional
            Boundary condition of the data. Defaults to "Periodic".
        even: bool, optional
            Flag indicating if the grid is evenly spaced. Defaults to True.
        grid_type:str, optional
            Type of grid, either "uniform" or "latlon". Defaults to "uniform".
        nbins: int, optional
            Number of bins for binning the data. Defaults to 10.

    Returns
    -------
        dict:
            Dictionary containing the requested structure functions and separation
            distances for the x- and y-direction (zonal and meridional, respectively).

    """
    # Initialize variables as NoneType
    SF_z = None
    SF_m = None
    SF_scalar_z = None
    SF_scalar_m = None
    SF_trad_velocity_z = None
    SF_trad_velocity_m = None
    SF_trad_scalar_z = None
    SF_trad_scalar_m = None
    SF_energy_prod_z = None
    SF_energy_prod_m = None
    advection = None

    # Define a list of separation distances to iterate over.
    # Periodic is half the length since the calculation will wrap the data.
    if boundary == "Periodic":
        sep_z = range(1, int(len(x) / 2))
        sep_m = range(1, int(len(y) / 2))
    else:
        sep_z = range(1, int(len(x) - 1))
        sep_m = range(1, int(len(y) - 1))

    # Initialize the separation distance arrays
    xd = np.zeros(len(sep_z) + 1)
    yd = np.zeros(len(sep_m) + 1)

    # Initialize the structure function arrays
    if skip_velocity_sf is False:
        SF_z = np.zeros(len(sep_z) + 1)
        SF_m = np.zeros(len(sep_m) + 1)


    if scalar is not None:
        SF_scalar_z = np.zeros(len(sep_z) + 1)
        SF_scalar_m = np.zeros(len(sep_m) + 1)

    if cal_shear:
        SF_energy_prod_z = np.zeros(len(sep_z) + 1)
        SF_energy_prod_m = np.zeros(len(sep_m) + 1)

    # establish keys for scalar SF
    if traditional_order > 0:
        trad_keys = ['SF_trad_'+ i for i in  ['u', 'v']] # , 'dudx', 'dudy', 'dvdx', 'dvdy'
        trad_keys.extend(['SF_trad_scalar']) if scalar is not None else None
    
        # allocate space for the traditional SFs
        SF_trad = dict()
        for kk in trad_keys:
            SF_trad[kk] = { 'z':np.zeros(len(sep_z) + 1), 
                            'm':np.zeros(len(sep_m) + 1) 
                        }
    # Calculate the advection and gradient components
    advection = calculate_advection(u, v, x, y, dx, dy, grid_type, scalar= scalar, gradients= cal_shear)

    def get_structure_per_lag_apply(lag_pair):

        down, right = lag_pair
        SF_dicts = calculate_structure_function(
            u,
            v,
            advection,
            down,
            right,
            skip_velocity_sf,
            scalar,
            traditional_order,
            boundary,
            cal_shear=cal_shear,
        )

        xroll = shift_array1d(x, shift_by=right, boundary=boundary)
        yroll = shift_array1d(y, shift_by=down, boundary=boundary)

        # Calculate separation distances in x and y
        xd_i, _ = calculate_separation_distances(
            x[right], y[right], xroll[right], yroll[right], grid_type
        )
        _, yd_i = calculate_separation_distances(
            x[down], y[down], xroll[down], yroll[down], grid_type
        )

        SF_dicts['right'] = right
        SF_dicts['down'] = down
        SF_dicts['xd_i'] = xd_i
        SF_dicts['yd_i'] = yd_i

        return SF_dicts


    if parallel:
        iter_list = zip(sep_m, sep_z)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use futures.map() to apply the function to each element of the list
            SF_collect = list(executor.map(get_structure_per_lag_apply, iter_list ))

    # linear version
    else:
        # Iterate over separations right and down\
        SF_collect = list()
        for lag_pair in zip(sep_m, sep_z):  # noqa: B905
            SF_collect.append(get_structure_per_lag_apply(lag_pair))

    # redistribute the results
    for SF_dicts in SF_collect:
        down, right = SF_dicts['down'], SF_dicts['right']

        if skip_velocity_sf is False:
            SF_z[right]                = SF_dicts["SF_adv_velocity_right"]
            SF_m[down]                 = SF_dicts["SF_adv_velocity_down"]

        if scalar is not None:
            SF_scalar_z[right]         = SF_dicts["SF_adv_scalar_right"]
            SF_scalar_m[down]          = SF_dicts["SF_adv_scalar_down"]

        if cal_shear:
            SF_energy_prod_z[right]     = SF_dicts["SF_energy_prod_right"]
            SF_energy_prod_m[down]      = SF_dicts["SF_energy_prod_down"]

        if traditional_order > 0:
            for kk in SF_trad.keys():
                SF_trad[kk]['z'][right] = SF_dicts[kk + '_right']
                SF_trad[kk]['m'][down]  = SF_dicts[kk + '_down']

        xd[right] = SF_dicts['xd_i']
        yd[down] = SF_dicts['yd_i']

    # Bin the data if the grid is uneven
    if even is False:
        if skip_velocity_sf is False:
            xd_bin, SF_z = bin_data(xd, SF_z, nbins)
            yd_bin, SF_m = bin_data(yd, SF_m, nbins)
            if traditional_order > 0:
                xd_bin, SF_trad_velocity_z = bin_data(xd, SF_trad_velocity_z, nbins)
                yd_bin, SF_trad_velocity_m = bin_data(yd, SF_trad_velocity_m, nbins)
        if scalar is not None:
            xd_bin, SF_scalar_z = bin_data(xd, SF_scalar_z, nbins)
            yd_bin, SF_scalar_m = bin_data(yd, SF_scalar_m, nbins)
            if traditional_order > 0:
                xd_bin, SF_trad_scalar_z = bin_data(xd, SF_trad_scalar_z, nbins)
                yd_bin, SF_trad_scalar_m = bin_data(yd, SF_trad_scalar_m, nbins)
        xd = xd_bin
        yd = yd_bin

    data = {
        "SF_advection_velocity_zonal": SF_z,
        "SF_advection_velocity_meridional": SF_m,
        "SF_advection_scalar_zonal": SF_scalar_z,
        "SF_advection_scalar_meridional": SF_scalar_m,

        "SF_energy_prod_zonal": SF_energy_prod_z,
        "SF_energy_prod_meridional": SF_energy_prod_m,

        # "SF_traditional_velocity_zonal": SF_trad_velocity_z,
        # "SF_traditional_velocity_meridional": SF_trad_velocity_m,
        # "SF_traditional_scalar_zonal": SF_trad_scalar_z,
        # "SF_traditional_scalar_meridional": SF_trad_scalar_m,

        "x_diff": xd,
        "y_diff": yd,
    }

    if traditional_order > 0:
        for kk in SF_trad.keys():
            data[kk+ "_zonal"]      = SF_trad[kk]['z']
            data[kk+ "_meridional"] = SF_trad[kk]['m']

    return data
