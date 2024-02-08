import numpy as np

from .shift_array2d import shift_array2d


def calculate_structure_function(  # noqa: D417
    u,
    v,
    adv_dict,
    down,
    right,
    skip_velocity_sf=False,
    scalar=None,
    traditional_order=0,
    boundary="Periodic",
    cal_shear = False
):
    """
    Calculate structure function, either advective or traditional.
    Supports velocity-based structure functions and scalar-based structure functions.

    Parameters
    ----------
        u: ndarray
            Array of u velocities.
        v: ndarray
            Array of v velocities.
        adv_dict: dict
            Dictionary containing the advection components and gradients.
        down: int
            Shift amount for downward shift. For periodic data should be less than half
            the column length and less than the column length for other boundary
            conditions.
        right: int
            Shift amount for rightward shift. For periodic data should be less than
            half the row length and less than the row length for other boundary
            conditions.
        skip_velocity_sf: bool, optional
            Whether to skip velocity-based structure function calculation.
            Defaults to False.
        scalar: ndarray, optional
            Array of scalar values. Defaults to None.
        traditional_order: int, optional
            Order for calculating traditional non-advective structure functions.
            If 0, no traditional structure functions are calculated. Defaults to 0.
        boundary: str, optional
            Boundary condition for shifting arrays. Defaults to "Periodic".
        cal_shear: bool, optional
            Flag used to calculate the shear structure function. Defaults to False.    

    Returns
    -------
        dict:
            A dictionary containing the advection velocity structure functions and
            scalar structure functions (if applicable).
            The dictionary has the following keys:
                'SF_adv_velocity_right': The advection velocity structure function in the
                right direction.
                'SF_adv_velocity_down': The advection velocity structure function in the
                down direction.
                'SF_trad_velocity_right': The traditional velocity structure function in
                the right direction (if traditional_order > 0).
                'SF_trad_velocity_down': The traditional velocity structure function in
                the down direction (if traditional_order > 0).
                'SF_adv_scalar_right': The scalar structure function in the right direction
                (if scalar is provided).
                'SF_adv_scalar_down': The scalar structure function in the down direction
                (if scalar is provided).
                'SF_trad_scalar_right': The traditional scalar structure function in the
                right direction (if scalar is provided and traditional_order > 0).
                'SF_trad_scalar_down': The traditional scalar structure function in the
                down direction (if scalar is provided and traditional_order > 0).
    """

    inputs = {
        "u": u,
        "v": v,
        "adv_eastward": adv_dict["eastward"] if "eastward" in adv_dict else None,
        "adv_northward": adv_dict["northward"] if "northward" in adv_dict else None,
        "scalar": scalar,
        "adv_scalar": adv_dict["scalar"] if "scalar" in adv_dict else None,
        "dudx": adv_dict["dudx"] if "dudx" in adv_dict else None,
        "dvdx": adv_dict["dvdx"] if "dvdx" in adv_dict else None,
        "dudy": adv_dict["dudy"] if "dudy" in adv_dict else None,
        "dvdy": adv_dict["dvdy"] if "dvdy" in adv_dict else None,
        "dsdx": adv_dict["dsdx"] if "dsdx" in adv_dict else None,
        "dsdy": adv_dict["dsdy"] if "dsdy" in adv_dict else None,
    }

    # create a list of traditional function keys
    traditional_keys = ['u', 'v']
    if scalar is not None:
        traditional_keys.append('scalar')


    if skip_velocity_sf is True:
        inputs.update(
            {
                "u": None,
                "v": None,
                "adv_eastward": None,
                "adv_northward": None,
            }
        )

    # Shift the input arrays by the down and right shift amounts
    shifted_inputs = {}
    for key, value in inputs.items():
        if value is not None:
            right_shift, down_shift = shift_array2d(
                inputs[key], shift_down=down, shift_right=right, boundary=boundary
            )

            shifted_inputs.update(
                {
                    key + "_right_shift": right_shift,
                    key + "_down_shift": down_shift,
                }
            )

    inputs.update(shifted_inputs)
    SF_dict = {}

    # interate over the right and down shifts
    for direction in ["right", "down"]:
        if skip_velocity_sf is False:
            adv_eastward = adv_dict["eastward"]
            adv_northward = adv_dict["northward"]
            SF_dict["SF_adv_velocity_" + direction] = np.nanmean(
                (inputs["adv_eastward_" + direction + "_shift"] - adv_eastward)
                * (inputs["u_" + direction + "_shift"] - u)
                + (inputs["adv_northward_" + direction + "_shift"] - adv_northward)
                * (inputs["v_" + direction + "_shift"] - v)
            )
        if (scalar is not None) & ("scalar" in adv_dict):
            adv_scalar = adv_dict["scalar"]
            SF_dict["SF_adv_scalar_" + direction] = np.nanmean(
                (inputs["adv_scalar_" + direction + "_shift"] - adv_scalar)
                * (inputs["scalar_" + direction + "_shift"] - scalar)
            )
        # else:
        #     raise ValueError("Scalar structure function requires scalar and adv_dict['scalar'] to be provided")

        # traditional structure functions for all scalar fields
        if traditional_order > 0:
            N = traditional_order
            for tkey in traditional_keys:
                SF_dict["SF_trad_"+tkey+"_" + direction] = np.nanmean(
                    (inputs[tkey+"_" + direction + "_shift"] - eval(tkey) ) ** N
                )

        if (cal_shear is True):
            if ("dudx" in inputs) & ("dudy" in inputs) & ("dvdx" in inputs) & ("dvdy" in inputs):
                
                dudx = adv_dict["dudx"]
                dudy = adv_dict["dudy"]
                dvdx = adv_dict["dvdx"]
                dvdy = adv_dict["dvdy"]

                SF_dict["SF_energy_prod_" + direction] = np.nanmean(
                    (inputs["u_" + direction + "_shift"] - u) ** 2 * (inputs["dudx_" + direction + "_shift"] - dudx ) +
                    (inputs["v_" + direction + "_shift"] - v) ** 2 * (inputs["dvdy_" + direction + "_shift"] - dvdy ) +
                    (inputs["u_" + direction + "_shift"] - u) * (inputs["v_" + direction + "_shift"] - v) * 
                    ( (inputs["dudy_" + direction + "_shift"] - dudy ) + (inputs["dvdx_" + direction + "_shift"] - dvdx ) )
                )
            else:
                raise ValueError("Shear structure function requires all gradients to be provided")
            
        if (cal_shear is True) & (scalar is not None):
            if ("dsdx" in inputs) & ("dsdy" in inputs):
                pass
                # check this calculation
                # SF_dict["SF_shear_scalar" + direction] = np.nanmean(
                #     (inputs["scalar_" + direction + "_shift"] - scalar) ** 2 * (inputs["dudx_" + direction + "_shift"] - dudx ) +
                #     (inputs["scalar_" + direction + "_shift"] - scalar) ** 2 * (inputs["dvdy_" + direction + "_shift"] - dvdy )
                # )
            else:
                raise ValueError("Shear structure function requires all gradients to be provided")

    return SF_dict
