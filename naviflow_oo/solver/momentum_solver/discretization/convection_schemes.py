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
    
    def calculate_flux_coefficients_vectorized(self, u_face, d_face, cell_peclet):
        """
        Vectorized version to calculate flux coefficients for multiple faces at once.
        
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
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_name(self):
        """Return the name of the discretization scheme."""
        return self.__class__.__name__


class PowerLawDiscretization(ConvectionDiscretization):
    """Power Law discretization scheme for convection terms."""
    
    def calculate_flux_coefficients(self, u_face, d_face, cell_peclet):
        """
        Calculate flux coefficients using the Power Law scheme.
        
        The Power Law scheme is an approximation of the exact solution to
        the one-dimensional convection-diffusion equation.
        """
        # Power law scheme: A(|P|) = max(0, (1-0.1|P|)^5)
        # where P is the cell Peclet number
        power_law_term = max(0, (1 - 0.1 * abs(cell_peclet))**5)
        
        # Combine diffusion (power law term * d_face) and upwind (max(0, -u_face))
        return power_law_term * d_face + max(0, -u_face)
    
    def calculate_flux_coefficients_vectorized(self, u_face, d_face, cell_peclet):
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
        
        # Combine diffusion and upwind
        return power_law_term * d_face + np.maximum(0, -u_face)
    
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
    
    def calculate_flux_coefficients_vectorized(self, u_face, d_face, cell_peclet):
        """
        Vectorized version to calculate flux coefficients using the first-order upwind scheme.
        
        Parameters:
        -----------
        u_face : ndarray
            Array of velocities at the faces
        d_face : float or ndarray
            Diffusion coefficient at the faces
        cell_peclet : ndarray
            Array of cell Peclet numbers (not used in upwind scheme)
            
        Returns:
        --------
        ndarray
            Array of flux coefficients
        """
        # Pure upwind scheme: only consider diffusion and upwind
        return d_face + np.maximum(0, -u_face)
    
    def get_name(self):
        return "First-order Upwind" 