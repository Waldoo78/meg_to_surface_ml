import numpy as np
import pyvista as pv

def show_comparison(orig_coords, recon_coords, triangles, error):
    """Show side by side comparison of original and reconstructed surfaces"""
    plotter = pv.Plotter(shape=(1, 2))
    
    # Convert triangles 
    if triangles.shape[1] == 3:
        pv_triangles = np.column_stack((np.full(len(triangles), 3), triangles))
    
    # Original surface
    plotter.subplot(0, 0)
    plotter.add_text("Original")
    mesh1 = pv.PolyData(orig_coords, pv_triangles)
    plotter.add_mesh(mesh1, color='lightblue', show_edges=True)
    
    # Reconstructed surface
    plotter.subplot(0, 1)
    plotter.add_text("Reconstruction")
    mesh2 = pv.PolyData(recon_coords, pv_triangles)
    plotter.add_mesh(mesh2, scalars=error, cmap="viridis", show_edges=True)
    plotter.add_scalar_bar("Error (mm)")
    
    plotter.show()

def show_surface(coords, triangles, scalars=None, title="Surface"):
    """Show a single surface with optional scalar data"""
    if triangles.shape[1] == 3:
        pv_triangles = np.column_stack((np.full(len(triangles), 3), triangles))
        
    plotter = pv.Plotter()
    mesh = pv.PolyData(coords, pv_triangles)
    
    plotter.add_mesh(
        mesh, 
        scalars=scalars, 
        cmap="viridis" if scalars is not None else None,
        color='lightblue' if scalars is None else None,
        show_edges=True
    )
    
    if scalars is not None:
        plotter.add_scalar_bar()
        
    plotter.add_text(title)
    plotter.show()