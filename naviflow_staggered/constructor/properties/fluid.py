"""
Fluid properties including viscosity, density, etc.
"""

class FluidProperties:
    """
    Class to store and manage fluid properties.
    """
    
    def __init__(self, density=1.0, viscosity=None, reynolds_number=None, characteristic_velocity=1.0, characteristic_length=1.0):
        """
        Initialize fluid properties.
        
        Parameters:
        -----------
        density : float
            Fluid density
        viscosity : float, optional
            Fluid viscosity. If not provided, calculated from Reynolds number.
        reynolds_number : float, optional
            Reynolds number. Required if viscosity is not provided.
        characteristic_velocity : float, optional
            Characteristic velocity for Reynolds number calculation
        characteristic_length : float, optional
            Characteristic length for Reynolds number calculation
        """
        self.density = density
        self.characteristic_velocity = characteristic_velocity
        self.characteristic_length = characteristic_length
        
        # Store Reynolds number if provided
        self.reynolds_number = reynolds_number
        
        # Calculate viscosity if not provided
        if viscosity is None:
            if reynolds_number is None:
                raise ValueError("Either viscosity or Reynolds number must be provided")
            self.viscosity = self.density * self.characteristic_velocity * self.characteristic_length / reynolds_number
        else:
            self.viscosity = viscosity
            # Calculate Reynolds number if not provided
            if reynolds_number is None:
                self.reynolds_number = self.density * self.characteristic_velocity * self.characteristic_length / self.viscosity
    
    def get_density(self):
        """Get fluid density."""
        return self.density
    
    def get_viscosity(self):
        """Get fluid viscosity."""
        return self.viscosity
    
    def get_reynolds_number(self):
        """Get Reynolds number."""
        return self.reynolds_number 