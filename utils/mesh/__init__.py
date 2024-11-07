# utils/mesh/__init__.py

from .Projection_onto_sphere import (
   get_spherical_projection,
   get_resampled_inner_surface
)

from .surface_generation import (
   generate_surface_from_coefficients,
   compute_surface_metrics  # Ajout de compute_surface_metrics
)

from .surface_preprocessing import (
   get_spherical_params,
   compute_template_projection,
   compute_coefficients
)

from .spherical_harmonics import (
   compute_Y,
   organize_coeffs,
   generate_surface_partial,
   generate_surface
)

from .visualization import (
   show_comparison,
   show_surface,
   show_spherical_projection  # Ajout de show_spherical_projection
)

__all__ = [
   # Spherical Harmonics functions
   'compute_Y',
   'organize_coeffs',
   'generate_surface_partial',
   'generate_surface',

   # Surface generation
   'generate_surface_from_coefficients',
   'compute_surface_metrics',  # Ajout

   # Surface preprocessing
   'get_spherical_params',
   'compute_template_projection',
   'compute_coefficients',

   # Projection onto sphere 
   'get_spherical_projection',
   'get_resampled_inner_surface',

   # Visualization
   'show_comparison',
   'show_surface',
   'show_spherical_projection'  # Ajout
]

