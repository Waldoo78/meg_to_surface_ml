import numpy as np
import pyvista as pv
from s3pipe.surface.surf import Surface
import s3pipe.surface.prop as sprop

def convert_triangles_to_pyvista(triangles):
    """Convert triangle format to PyVista format"""
    return triangles if triangles.shape[1] == 4 else np.column_stack((np.full(len(triangles), 3), triangles))

def setup_plotter_camera(plotter, coords):
    """Setup standard camera position based on surface bounds"""
    center = np.mean(coords, axis=0)
    max_bound = np.max(np.ptp(coords, axis=0))
    plotter.camera_position = [center + [0, 0, max_bound * 2], center, [0, 1, 0]]

def create_mesh_with_scalars(coords, triangles, scalar_data, cmap, title):
    """Helper to create mesh with consistent settings"""
    mesh = pv.PolyData(coords, convert_triangles_to_pyvista(triangles))
    return {
        'mesh': mesh,
        'scalars': scalar_data,
        'cmap': cmap,
        'show_edges': True,
        'scalar_bar_args': {'title': title}
    }

def show_comparison(orig_coords, recon_coords, triangles, metrics, title=None):  # title devient optionnel
   """Show side by side comparison with metrics"""
   plotter = pv.Plotter(shape=(1, 2))
   
   # Original surface
   plotter.subplot(0, 0)
   plotter.add_text("Original Surface", position='upper_edge', font_size=12)
   mesh_args = create_mesh_with_scalars(
       orig_coords, triangles,
       metrics['raw_metrics']['reference_curvature'],
       'coolwarm', 'Mean Curvature'
   )
   plotter.add_mesh(**mesh_args)
   setup_plotter_camera(plotter, orig_coords)
   
   # Reconstructed surface
   plotter.subplot(0, 1)
   plotter.add_text("Reconstructed Surface", position='upper_edge', font_size=12)
   mesh_args = create_mesh_with_scalars(
       recon_coords, triangles,
       metrics['raw_metrics']['error'],
       'viridis', 'Error (mm)'
   )
   plotter.add_mesh(**mesh_args)
   setup_plotter_camera(plotter, recon_coords)
   
   # Add metrics
   stats = metrics['distance_stats']
   stats_text = (
       f"Mean Error: {stats['mean_error']:.2f} mm\n"
       f"Max Error: {stats['max_error']:.2f} mm\n"
       f"Hausdorff: {stats['error_hausdorff']:.2f} mm\n"
       f"Curvature Diff: {metrics['shape_stats']['mean_curvature_diff']:.2f}"
   )
   plotter.add_text(stats_text, position=(0.7, 0.02), font_size=12)
   
   plotter.link_views()
   plotter.show()

def show_surface(coords, triangles, title="Surface Visualization"):
    """Show a single surface"""
    plotter = pv.Plotter()
    mesh_args = create_mesh_with_scalars(coords, triangles, None, "viridis", "")
    plotter.add_mesh(**mesh_args)
    setup_plotter_camera(plotter, coords)
    plotter.add_text(title, position='upper_edge', font_size=16)
    plotter.show()

def show_spherical_projection(orig_coords, sphere_coords, triangles, title="Spherical Projection"):
    """Show original surface and its spherical projection"""
    pv_triangles = convert_triangles_to_pyvista(triangles)
    
    # Compute areas and distortion
    orig_areas = sprop.computeVertexArea(Surface(orig_coords, pv_triangles))
    sphere_areas = sprop.computeVertexArea(Surface(sphere_coords, pv_triangles))
    area_ratio = sphere_areas / orig_areas
    
    plotter = pv.Plotter(shape=(1, 2))
    
    # Original surface
    plotter.subplot(0, 0)
    plotter.add_text("Original Surface")
    mesh_args = create_mesh_with_scalars(orig_coords, triangles, orig_areas, "viridis", "Vertex Area")
    plotter.add_mesh(**mesh_args)
    setup_plotter_camera(plotter, orig_coords)
    
    # Sphere projection
    plotter.subplot(0, 1)
    plotter.add_text("Spherical Projection")
    mesh_args = create_mesh_with_scalars(sphere_coords, triangles, area_ratio, "coolwarm", "Area Distortion")
    plotter.add_mesh(**mesh_args)
    setup_plotter_camera(plotter, sphere_coords)
    
    # Add title and stats
    plotter.add_text(title, position='upper_edge', font_size=16)
    stats_text = (
        f"Mean Area Distortion: {np.mean(area_ratio):.2f}\n"
        f"Max Area Distortion: {np.max(area_ratio):.2f}\n"
        f"Std Area Distortion: {np.std(area_ratio):.2f}"
    )
    plotter.add_text(stats_text, position='lower_right', font_size=12)
    
    plotter.link_views()
    plotter.show()