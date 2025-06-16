import io
import os
import numpy as np
from astropy import units as U, constants as C
from astropy.coordinates import ICRS
from .sph_source import SPHSource
from ..sph_kernels import _CubicSplineKernel, find_fwhm


def api_get(path, params=None, api_key=None):
    """
    Make a request to the TNG web API service.

    Parameters
    ----------
    path : str
        The request to submit to the API.

    params : dict, optional
        Additional options for the API request. (Default: ``None``)

    api_key : str
        API key to authenticate to the TNG web API service. (Default: ``None``)

    Returns
    -------
    out : str
        Response from the API, a JSON-encoded string.
    """
    import requests

    baseUrl = "http://www.tng-project.org/api/"
    r = requests.get(f"{baseUrl}/{path}", params=params, headers={"api-key": api_key})
    r.raise_for_status()
    if r.headers["content-type"] == "application/json":
        return r.json()
    return r


def cutout_file(simulation, snapNum, haloID):
    """
    Helper to generate a string identifying a cutout file.

    Parameters
    ----------
    simulation : str
        Identifier of the simulation.

    snapNum : int
        Snapshot identifier.

    haloID : int
        Halo identifier.

    Returns
    -------
    out : str
        A string to use for a cutout file.
    """
    return f"martini-cutout-{simulation}-{snapNum}-{haloID}.hdf5"


class AurigaSource(SPHSource):
    """
    Class abstracting HI sources for use with Auriga simulations.

    To be updated.

    Parameters
    ----------
    simulation : str
        Simulation identifier string, for example ``"TNG100-1"``, see
        https://www.tng-project.org/data/docs/background/

    snapNum : int
        Snapshot number. In TNG, snapshot 99 is the final output. Note that
        if a 'mini' snapshot (see
        http://www.tng-project.org/data/docs/specifications/#sec1a) is selected then
        some additional approximations are used.

    subID : int
        Subhalo ID of the target object. Note that all particles in the FOF
        group to which the subhalo belongs are used to construct the data cube.
        This avoids strange 'holes' at the locations of other subhaloes in the
        same group, and gives a more realistic treatment of foreground and
        background emission local to the source. An object of interest could be
        found using https://www.tng-project.org/data/search/, for instance. The
        "ID" column in the search results on that page is the subID.

    api_key: str, optional
        Use of the TNG web API requires an API key: login at
        https://www.tng-project.org/users/login/ or register at
        https://www.tng-project.org/users/register/ then obtain your API
        key from https://www.tng-project.org/users/profile/ and provide as a string. An
        API key is not required if logged into the TNG JupyterLab. (Default: ``None``)

    cutout_dir: str, optional
        Ignored if running on the TNG JupyterLab. Directory in which to search for and
        save cutout files (see documentation at
        https://www.tng-project.org/data/docs/api/ for a description of cutouts). Cutout
        filenames are enforced by MARTINI. If `cutout_dir==None` (the default), then the
        data will always be downloaded. If a `cutout_dir` is provided, it will first be
        searched for the required data. If the data are found, the local copy is used,
        otherwise the data are downloaded and a local copy is saved in `cutout_dir` for
        future use. (Default: ``None``)

    distance : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of length.
        Source distance, also used to set the velocity offset via Hubble's law.
        (Default: ``3 * U.Mpc``)

    vpeculiar : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of velocity.
        Source peculiar velocity along the direction to the source centre.
        (Default: ``0 * U.km * U.s**-1``)

    rotation : dict, optional
        Must have a single key, which must be one of ``axis_angle``, ``rotmat`` or
        ``L_coords``. Note that the 'y-z' plane will be the one eventually placed in the
        plane of the "sky". The corresponding value must be:

        - ``axis_angle`` : 2-tuple, first element one of 'x', 'y', 'z' for the \
        axis to rotate about, second element a :class:`~astropy.units.Quantity` with \
        dimensions of angle, indicating the angle to rotate through.
        - ``rotmat`` : A (3, 3) :class:`~numpy.ndarray` specifying a rotation.
        - ``L_coords`` : A 2-tuple containing an inclination and an azimuthal \
        angle (both :class:`~astropy.units.Quantity` instances with dimensions of \
        angle). The routine will first attempt to identify a preferred plane \
        based on the angular momenta of the central 1/3 of particles in the \
        source. This plane will then be rotated to lie in the plane of the \
        "sky" ('y-z'), rotated by the azimuthal angle about its angular \
        momentum pole (rotation about 'x'), and inclined (rotation about \
        'y'). A 3-tuple may be provided instead, in which case the third \
        value specifies the position angle on the sky (second rotation about 'x'). \
        The default position angle is 270 degrees.

        (Default: ``np.eye(3)``)

    ra : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Right ascension for the source centroid. (Default: ``0 * U.deg``)

    dec : ~astropy.units.Quantity, optional
        :class:`~astropy.units.Quantity`, with dimensions of angle.
        Declination for the source centroid. (Default: ``0 * U.deg``)

    coordinate_frame : ~astropy.coordinates.builtin_frames.baseradec.BaseRADecFrame, \
    optional
        The coordinate frame assumed in converting particle coordinates to RA and Dec, and
        for transforming coordinates and velocities to the data cube frame. The frame
        needs to have a well-defined velocity as well as spatial origin. Recommended
        frames are :class:`~astropy.coordinates.GCRS`, :class:`~astropy.coordinates.ICRS`,
        :class:`~astropy.coordinates.HCRS`, :class:`~astropy.coordinates.LSRK`,
        :class:`~astropy.coordinates.LSRD` or :class:`~astropy.coordinates.LSR`. The frame
        should be passed initialized, e.g. ``ICRS()`` (not just ``ICRS``).
        (Default: ``astropy.coordinates.ICRS()``)
    """

    def __init__(
        self,
        outputdir,
        snap_num,
        aperture=50.0 * U.kpc,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        GK11=False,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        coordinate_frame=ICRS(),
    ):
        # optional dependencies for this source class
        import auriga_public as ap
        # import h5py
        # from Hdecompose.atomic_frac import atomic_frac

        parttype=0
        fields_g = (
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "Density",
            "Volume",
            "Coordinates",
            "GFM_Metals",
            "NeutralHydrogenAbundance",
            "StarFormationRate"
        )
        snapobj = ap.snapshot.load_snapshot(snap_num, parttype, loadlist=fields_g,
                                            snappath=outputdir, verbose=False)
        subobj = ap.subhalos.subfind(snap_num, directory=outputdir,
                                    loadlist=['SubhaloPos', 'Group_R_Crit200'])
        snapobj = ap.util.CentreOnHalo(snapobj, subobj.data['SubhaloPos'][0])
        self.bulk_velocity = ap.util.remove_bulk_velocity(snapobj, idx=None,
                                         radialcut=0.1*subobj.data['Group_R_Crit200'][0])

        a = snapobj.header["Time"]
        self.a = a
        self.bulk_velocity *= np.sqrt(a)
        # z = snapobj.header["Redshift"]
        h = snapobj.header["HubbleParam"]
        snapobj = ap.util.apply_mask(snapobj, radialcut=(aperture << a / h * U.Mpc).value)

        X_H_g = snapobj.data["GFM_Metals"][:,0]
        xe_g = snapobj.data["ElectronAbundance"]
        rho_g = snapobj.data["Density"] << (1e10 / h * U.Msun * np.power(a / h * U.Mpc, -3))
        u_g = snapobj.data["InternalEnergy"] << (U.km/U.s)**2
        mu_g = 4 / (1 + 3 * X_H_g + 4 * X_H_g * xe_g) << C.m_p
        m_g = snapobj.data["Masses"] << (1e10 / h * U.Msun)
        # cast to float64 to avoid underflow error
        nH_g = rho_g * X_H_g / C.m_p << U.cm**-3
        fneutral_g = snapobj.data["NeutralHydrogenAbundance"].copy()
        gamma = 5.0 / 3.0
        # cold
        mu_c = 4 / (1 + 3 * X_H_g) << C.m_p
        u_c = C.k_B * (1e3<<U.K) / (mu_c * (gamma - 1.)) << (U.km/U.s)**2
        del mu_c
        # hot
        mu_h = 4 / (3 + 5 * X_H_g) << C.m_p # He fully ionized
        T_h = (1e3
               + 5.73e7 / (1 +
                           573*np.maximum(1.,
                                          nH_g.to_value(U.cm**-3)/0.13
                                          )**-0.8)
               ) << U.K # SH03; Stevens 19
        u_h = C.k_B * T_h / (mu_h * (gamma - 1.)) << (U.km/U.s)**2
        del mu_h
        sfr_g = snapobj.data["StarFormationRate"] << U.Msun / U.yr
        possfr_mask = sfr_g > 0
        u_h_pos = u_h[possfr_mask]
        fneutral_g[possfr_mask] = (1. / (u_h_pos - u_c[possfr_mask]) \
                                   * (u_h_pos - u_g[possfr_mask]) << 1).value
        del u_h_pos, sfr_g, possfr_mask, u_c, u_h

        if GK11:
            raise NotImplementedError
        else:
            # This is partial pressure; see M17
            P_g = (gamma - 1.) / C.k_B * u_g * fneutral_g * rho_g << U.K/U.cm**3
            fatomic_g = 1. / (1. +
                (1. / (1.7e4 << U.K/U.cm**3) * P_g) ** 0.8
            )
            T_g = (gamma - 1.) / C.k_B * u_g * mu_g << U.K
            del P_g, rho_g, u_g
            # T_g is only used for velocity dispersion later
        mHI_g = m_g * X_H_g * fatomic_g * fneutral_g << U.Msun
        del m_g, X_H_g, fatomic_g, fneutral_g
        xyz_g = snapobj.data["Coordinates"] * (a / h) << U.Mpc
        vxyz_g = snapobj.data["Velocities"] * np.sqrt(a) << U.km / U.s
        hsm_g = np.cbrt(snapobj.data["Volume"]) << U.kpc
        hsm_g *= np.cbrt(3.0 / 4.0 / np.pi) * (a/h) * U.Mpc.to(U.kpc) # r_cell
        # hsm_g has in mind a cubic spline that =0 at r=h, I think
        hsm_g *= 2.5 * find_fwhm(_CubicSplineKernel().kernel)

        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
            coordinate_frame=coordinate_frame
        )
        return
