import numpy as np


def calculate_advection(  # noqa: D417
    u,
    v,
    x,
    y,
    dx=None,
    dy=None,
    grid_type="uniform",
    scalar=None,
    gradients = False,
):
    """
    Calculate the advection for a velocity field or scalar field. The velocity field
    will return advection components in the eastward and northward directions.
    The scalar field will return the scalar advection. Defaults to advection for
    velocity field. If the velocity advection is skipped or a scalar field is not
    provided, the relevant dictionary key will return None.

    Parameters
    ----------
        u: ndarray
            The u-component of velocity.
        v: ndarray
            The v-component of velocity.
        x: ndarray
            The x-coordinates of the grid.
        y: ndarray
            The y-coordinates of the grid.
        dx: float or ndarray, optional
            The grid spacing in the x-direction. Defaults to None.
        dy: float or ndarray, optional
            The grid spacing in the y-direction. Defaults to None.
        grid_type: str, optional
            The type of grid. Defaults to "uniform".
        scalar: ndarray, optional
            Array of scalar values. Defaults to None.
        gradients: bool, optional
            Flag used to calculate the gradients of the velocity field. Defaults to False.

    Returns
    -------
        tuple or ndarray:
            A tuple of advection components (eastward_advection,
            northward_advection) if scalar is not provided, otherwise returns an ndarray
            of scalar advection.
    """
    # calculate the gradients depending on the grid type
    if grid_type == "latlon":
        xcoords = dx.cumsum()
        ycoords = dy.cumsum()

        if scalar is not None:
            dsdy, dsdx = np.gradient(scalar, xcoords, ycoords, axis=(0, 1))
        #else:
        dudy, dudx = np.gradient(u, xcoords, ycoords, axis=(0, 1))
        dvdy, dvdx = np.gradient(v, xcoords, ycoords, axis=(0, 1))

    else:
        dx = np.abs(x[0] - x[1])
        dy = np.abs(y[0] - y[1])

        if scalar is not None:
            dsdy, dsdx = np.gradient(scalar, dx, dy, axis=(0, 1))
        #else:
        dudy, dudx = np.gradient(u, dx, dy, axis=(0, 1))
        dvdy, dvdx = np.gradient(v, dx, dy, axis=(0, 1))

    # return the advection components
    return_dict = {}
    # if scalar is None:
    return_dict['eastward']  = u * dudx + v * dudy
    return_dict['northward'] = u * dvdx + v * dvdy

    if scalar is not None:
        return_dict['scalar'] = u * dsdx + v * dsdy

    if gradients:
        return_dict['dudx'] = dudx
        return_dict['dudy'] = dudy
        return_dict['dvdx'] = dvdx
        return_dict['dvdy'] = dvdy
        if scalar is not None:
            return_dict['dsdx'] = dsdx
            return_dict['dsdy'] = dsdy

    return return_dict
