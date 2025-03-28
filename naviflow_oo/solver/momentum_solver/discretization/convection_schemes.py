"""
Convection discretization schemes for momentum equations.
This module contains different discretization schemes for the convective terms
in the momentum equations, such as Power Law, QUICK, etc.
"""
import numpy as np

class ConvectionDiscretization:
    """Base class for convection discretization schemes."""
    
    def calculate_flux_coefficients(self, u_face, d_face, cell_peclet):
        """
        Calculate flux coefficients based on the discretization scheme.
        
        Parameters:
        -----------
        u_face : float
            Velocity at the face
        d_face : float
            Diffusion coefficient at the face
        cell_peclet : float
            Cell Peclet number (ratio of convection to diffusion)
            
        Returns:
        --------
        float
            Flux coefficient for the face
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def get_name(self):
        """Return the name of the discretization scheme."""
        return self.__class__.__name__


class PowerLawDiscretization(ConvectionDiscretization):
    """Power Law discretization scheme for convection terms."""
    

    def calculate_flux_coefficients(self, u_face, d_face, cell_peclet):
        """
        Vectorized version to calculate flux coefficients using the Power Law scheme.
        
        Parameters:
        -----------
        u_face : ndarray
            Array of velocities at the faces
        d_face : float or ndarray
            Diffusion coefficient at the faces
        cell_peclet : ndarray
            Array of cell Peclet numbers
            
        Returns:
        --------
        ndarray
            Array of flux coefficients
        """
        # Power law scheme: A(|P|) = max(0, (1-0.1|P|)^5)
        abs_peclet = np.abs(cell_peclet)
        power_law_term = np.maximum(0, (1 - 0.1 * abs_peclet)**5)
        
        # Match the original sign convention for upwind terms
        if isinstance(u_face, np.ndarray):
            upwind_term = np.where(u_face >= 0, np.maximum(0, -u_face), np.maximum(0, u_face))
        else:
            upwind_term = np.maximum(0, -u_face if u_face >= 0 else u_face)
        
        return power_law_term * d_face + upwind_term
    
    def get_name(self):
        return "Power Law"


class UpwindDiscretization(ConvectionDiscretization):
    """First-order upwind discretization scheme for convection terms."""
    
    def calculate_flux_coefficients(self, u_face, d_face, cell_peclet):
        """
        Calculate flux coefficients using the first-order upwind scheme.
        
        The upwind scheme takes into account the flow direction by using
        the value from the upstream cell.
        """
        # Pure upwind scheme: only consider diffusion and upwind
        return d_face + max(0, -u_face)

    def get_name(self):
        return "First-order Upwind"


class QuickDiscretization(ConvectionDiscretization):
    """QUICK (Quadratic Upstream Interpolation for Convective Kinematics) discretization scheme."""
    

    def calculate_flux_coefficients(self, u_face, d_face, cell_peclet):
        """
        Vectorized version of the QUICK scheme calculation.
        
        Parameters:
        -----------
        u_face : ndarray
            Array of velocities at the faces
        d_face : float or ndarray
            Diffusion coefficient at the faces
        cell_peclet : ndarray
            Array of cell Peclet numbers
            
        Returns:
        --------
        ndarray
            Array of flux coefficients
        """
        # Upwind component
        upwind_component = np.maximum(0, -u_face)
        
        # Central differencing component approximation
        abs_peclet = np.abs(cell_peclet)
        central_weight = np.maximum(0, 1.0 - 0.5 * abs_peclet)
        central_component = 0.5 * d_face * central_weight
        
        # QUICK blending based on Peclet number
        # Create a mask for where Peclet number is less than 10
        moderate_peclet_mask = abs_peclet < 10
        
        # Initialize with the upwind scheme
        result = np.full_like(u_face, d_face) + upwind_component
        
        # Apply the QUICK blend only where Peclet number is moderate
        result[moderate_peclet_mask] += central_component[moderate_peclet_mask]
        
        return result
    
    def get_name(self):
        return "QUICK" 