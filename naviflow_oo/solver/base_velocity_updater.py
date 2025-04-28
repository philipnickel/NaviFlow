from abc import ABC, abstractmethod

class VelocityUpdater(ABC):
    """
    Abstract base class defining the interface for velocity updaters.
    Implementations should be mesh-agnostic and work with 1D arrays.
    """
    
    @abstractmethod
    def setup(self, mesh, fluid_properties):
        """
        Set up the velocity updater with mesh and fluid properties.
        
        Parameters
        ----------
        mesh : Mesh
            The computational mesh
        fluid_properties : FluidProperties
            The fluid properties
        """
        pass
    
    @abstractmethod
    def correct_velocity(self, u_star, v_star, p_prime, d_u, d_v):
        """
        Correct the predicted velocities using the pressure correction.
        
        Parameters
        ----------
        u_star, v_star : ndarray
            Predicted velocity fields (1D arrays)
        p_prime : ndarray
            Pressure correction field (1D array)
        d_u, d_v : ndarray
            Diagonal coefficients from momentum matrix (1D arrays)
            
        Returns
        -------
        tuple
            (u_new, v_new) corrected velocity fields as 1D arrays
        """
        pass
    
    @abstractmethod
    def calculate_mass_flux(self, u, v, p=None, d_u=None, d_v=None):
        """
        Calculate the mass flux through cell faces.
        Optionally applies Rhie-Chow correction if pressure and diagonal coefficients provided.
        
        Parameters
        ----------
        u, v : ndarray
            Velocity fields as 1D arrays
        p : ndarray, optional
            Pressure field as a 1D array (for Rhie-Chow correction)
        d_u, d_v : ndarray, optional
            Diagonal coefficients from momentum matrix (for Rhie-Chow correction)
            
        Returns
        -------
        ndarray
            Mass flux through each face
        """
        pass 