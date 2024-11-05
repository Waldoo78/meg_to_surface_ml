import os
import tempfile
import numpy as np
import pyvista as pv
from utils.file_manip import vtk_processing
from utils.file_manip import Matlab_to_array 

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def spherical_projection_40962(mesh_data, hemisphere='lh'):
    """Projects a mesh onto a sphere at 40962 vertices using S3MAP."""
    if hemisphere not in ['lh', 'rh']:
        raise ValueError("hemisphere must be 'lh' or 'rh'")
   
    coords, triangles = mesh_data
    print(f"Input mesh size: {len(coords)} vertices")
    
    # Setup temporary files
    temp_dir = tempfile.gettempdir()
    input_vtk = os.path.join(temp_dir, f"{hemisphere}.brain.vtk")
    sphere_vtk = input_vtk.replace('.vtk', '.inflated.SIP.10242moved.RespSphe.40962moved.vtk')
    target_vtk = input_vtk.replace('.vtk', '.inflated.SIP.10242moved.RespInner.vtk')
    
    # Save input mesh
    vtk_processing.save_to_vtk(coords, triangles, input_vtk)
    
    # Run S3MAP projection
    abs_path = os.path.abspath(input_vtk)
    s3map_script = os.path.join(PROJECT_ROOT, "utils", "mesh", "S3MAP-main", "s3all.py")
    cmd = f"python {s3map_script} -i {abs_path} --save_interim_results True --device CPU"
    print(f"Executing: {cmd}")  # Pour debug
    os.system(cmd)

    # Load results
    if not os.path.exists(sphere_vtk) or not os.path.exists(target_vtk):
        print(f"Looking for files:\n{sphere_vtk}\n{target_vtk}")
        raise FileNotFoundError("S3MAP output files not found")

    sphere_coords, sphere_triangles = vtk_processing.vtk_mesh_to_array(sphere_vtk)
    target_coords, target_triangles = vtk_processing.vtk_mesh_to_array(target_vtk)
    
    return sphere_coords, sphere_triangles, target_coords, target_triangles


def main_test():
    
    def visualize_mesh(coords, tris, title):
        """Visualizes a mesh using PyVista
        
        Args:
            coords (ndarray): Vertex coordinates
            tris (ndarray): Triangle indices
            title (str): Plot title
        """
        plotter = pv.Plotter()
        mesh = pv.PolyData(coords, np.hstack((np.full((len(tris), 1), 3), tris)))
        plotter.add_mesh(mesh, show_edges=True, color='white')
        plotter.add_text(title, font_size=12)
        plotter.show()
    
    
    # Load test data
    faces = Matlab_to_array.load_faces(r"C:\Users\wbou2\Documents\meg_to_surface_ml\src\data\Faces_1.mat")
    vertices = Matlab_to_array.load_vertices(r"C:\Users\wbou2\Documents\meg_to_surface_ml\src\data\Vertices_1.mat")
    
    # Extract left hemisphere
    mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]))
    center = np.mean(vertices, axis=0)
    normal = [0, -1, 0]
    hemisphere_mesh = mesh.clip(normal=normal, origin=center)
    
    lh_faces = np.array(hemisphere_mesh.faces.reshape(-1, 4)[:, 1:4], dtype=np.int32)
    lh_coords = np.array(hemisphere_mesh.points, dtype=np.float64)
    
    # Perform projection
    lh_sphere_coords, lh_sphere_tris, lh_40k_coords, lh_40k_tris = \
        spherical_projection_40962((lh_coords, lh_faces), hemisphere='lh')
    
    # Visualize results
    visualize_mesh(lh_coords, lh_faces, f"Original mesh ({len(lh_coords)} vertices)")
    visualize_mesh(lh_40k_coords, lh_40k_tris, "Resampled mesh (40962 vertices)")
    visualize_mesh(lh_sphere_coords, lh_sphere_tris, "Spherical projection (40962 vertices)")
    
    # Print summary
    print("\nMesh sizes:")
    print(f"Original mesh: {len(lh_coords)} vertices")
    print(f"Resampled mesh: {len(lh_40k_coords)} vertices")
    print(f"Spherical projection: {len(lh_sphere_coords)} vertices")

if __name__ == '__main__':
    main_test()