import numpy as np
import pyvista as pv
from s3pipe.surface.surf import Surface
import s3pipe.surface.prop as sprop

def convert_triangles_to_pyvista(triangles):
    """Convert triangle format to PyVista format
    
    Args:
        triangles (ndarray): Nx3 or Nx4 triangle array
        
    Returns:
        ndarray: Nx4 triangle array with count prefix
    """
    if triangles.shape[1] == 3:
        return np.column_stack((np.full(len(triangles), 3), triangles))
    return triangles

def setup_plotter_camera(plotter, coords):
    """Setup standard camera position based on surface bounds
    
    Args:
        plotter (pyvista.Plotter): PyVista plotter
        coords (ndarray): Surface coordinates
    """
    center = np.mean(coords, axis=0)
    bounds = np.ptp(coords, axis=0)
    max_bound = np.max(bounds)
    plotter.camera_position = [
        center + [0, 0, max_bound * 2],  # Camera position
        center,  # Focal point
        [0, 1, 0]  # Up vector
    ]
def show_comparison(orig_coords, recon_coords, triangles, metrics, title="Surface Comparison"):
    """Show side by side comparison with metrics"""
    # Prepare triangles
    pv_triangles = convert_triangles_to_pyvista(triangles)
    
    # Create plotter with side-by-side view
    plotter = pv.Plotter(shape=(1, 2))
    
    # Original surface with curvature
    plotter.subplot(0, 0)
    plotter.add_text("Original Surface (Mean Curvature)")
    mesh1 = pv.PolyData(orig_coords, pv_triangles)
    plotter.add_mesh(mesh1, 
                    scalars=metrics['raw_metrics']['reference_curvature'],
                    cmap='coolwarm',
                    show_edges=True,
                    scalar_bar_args={'title': 'Mean Curvature'})
    setup_plotter_camera(plotter, orig_coords)
    
    # Reconstructed surface with error
    plotter.subplot(0, 1)
    plotter.add_text("Reconstructed Surface (Error)")
    mesh2 = pv.PolyData(recon_coords, pv_triangles)
    plotter.add_mesh(mesh2,  
                    scalars=metrics['raw_metrics']['error'],
                    cmap='viridis',
                    show_edges=True,
                    scalar_bar_args={'title': 'Error (mm)'})
    setup_plotter_camera(plotter, recon_coords)
    
    # Add title and metrics summary
    plotter.add_text(title, position='upper_edge', font_size=16)
    
    # Access metrics correctly from the dictionary
    stats_text = (
        f"Mean Error: {metrics['distance_stats']['mean_error']:.2f} mm\n"
        f"Max Error: {metrics['distance_stats']['max_error']:.2f} mm\n"
        f"Hausdorff: {metrics['distance_stats']['error_hausdorff']:.2f} mm\n"
        f"Curvature Diff: {metrics['shape_stats']['mean_curvature_diff']:.2f}"
    )
    plotter.add_text(stats_text, position='lower_right', font_size=12)
    
    plotter.link_views()
    plotter.show()
def show_surface(coords, triangles, title="Surface Visualization"):
    """Show a single surface 
    
    Args:
        coords (ndarray): Surface coordinates (N_points, 3)
        triangles (ndarray): Surface triangulation (N_tris, 3/4)
        title (str): Title for the visualization
    """
    # Prepare triangles
    pv_triangles = convert_triangles_to_pyvista(triangles)
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Create mesh and add to plotter
    mesh = pv.PolyData(coords, pv_triangles)
    plotter.add_mesh(
        mesh,
        cmap="viridis",
        show_edges=True,
    )
    
    # Setup camera and add title
    setup_plotter_camera(plotter, coords)
    plotter.add_text(title, position='upper_edge', font_size=16)
    
    plotter.show()

def show_spherical_projection(orig_coords, sphere_coords, triangles, title="Spherical Projection"):
    """Show original surface and its spherical projection
    
    Args:
        orig_coords (ndarray): Original surface coordinates
        sphere_coords (ndarray): Projected spherical coordinates
        triangles (ndarray): Surface triangulation
        title (str): Title for the visualization
    """
    # Prepare triangles
    pv_triangles = convert_triangles_to_pyvista(triangles)
    
    # Create surfaces
    orig_surf = Surface(orig_coords, pv_triangles)
    sphere_surf = Surface(sphere_coords, pv_triangles)
    
    # Compute distortion metric
    orig_areas = sprop.computeVertexArea(orig_surf)
    sphere_areas = sprop.computeVertexArea(sphere_surf)
    area_ratio = sphere_areas / orig_areas
    
    # Create visualization
    plotter = pv.Plotter(shape=(1, 2))
    
    # Original surface
    plotter.subplot(0, 0)
    plotter.add_text("Original Surface")
    mesh1 = pv.PolyData(orig_coords, pv_triangles)
    plotter.add_mesh(mesh1, scalars=orig_areas, cmap="viridis", 
                    show_edges=True, scalar_bar_args={'title': 'Vertex Area'})
    setup_plotter_camera(plotter, orig_coords)
    
    # Spherical projection
    plotter.subplot(0, 1)
    plotter.add_text("Spherical Projection")
    mesh2 = pv.PolyData(sphere_coords, pv_triangles)
    plotter.add_mesh(mesh2, scalars=area_ratio, cmap="coolwarm", 
                    show_edges=True, scalar_bar_args={'title': 'Area Distortion'})
    setup_plotter_camera(plotter, sphere_coords)
    
    # Add title and statistics
    plotter.add_text(title, position='upper_edge', font_size=16)
    stats_text = f"Mean Area Distortion: {np.mean(area_ratio):.2f}\n"
    stats_text += f"Max Area Distortion: {np.max(area_ratio):.2f}\n"
    stats_text += f"Std Area Distortion: {np.std(area_ratio):.2f}"
    plotter.add_text(stats_text, position='lower_right', font_size=12)
    
    plotter.link_views()
    plotter.show()