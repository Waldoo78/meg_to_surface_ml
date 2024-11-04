# utils/file_manip/__init__.py
from .vtk_processing import (
    vtk_mesh_to_array,
    save_to_vtk
)

from .Matlab_to_array import (
    load_faces,
    load_vertices
)

__all__ = [
    # VTK Functions
    'vtk_mesh_to_array',
    'save_to_vtk',
    
    # Matlab conversion functions
    
    "load_faces"
    "load_vertices"
]