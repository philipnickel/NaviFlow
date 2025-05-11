from naviflow_collocated.discretization.convection.power_law import (
    compute_convective_flux_power_law,
)
from naviflow_collocated.discretization.convection.upwind import (
    compute_convective_stencil_upwind,
)
import numba


@numba.njit(inline="always")
def compute_convective_flux(f, phi, grad_phi, mesh, rho, u, scheme="power_law"):
    if scheme == "power_law":
        # Convert the scalar return to a tuple format matching upwind's return format
        flux = compute_convective_flux_power_law(f, phi, grad_phi, mesh, rho, u)
        # Return a consistent tuple of (a_P, a_N, b_corr)
        # For power law, we don't separate the flux into coefficients
        # so we return (0, 0, flux) to be consistent
        return 0.0, 0.0, flux
    elif scheme == "upwind":
        return compute_convective_stencil_upwind(f, phi, grad_phi, mesh, rho, u)
    # Ensure we always return a tuple
    return 0.0, 0.0, 0.0
