import numpy as np
import pyvista as pv
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import surface
from Matlab_to_array import load_faces, load_vertices
from Projection_onto_sphere import spherical_projection_40962
from surface_reconstruction import process_surface, visualize_results

def extract_hemisphere(mesh, hemisphere='left'):
    """
    Extract left or right hemisphere from a brain mesh.
    Y positif = gauche
    Y négatif = droit
    """
    points = mesh.points
    
    if hemisphere == 'left':
        indices = points[:, 1] >= 0  # positive y for left
    else:
        indices = points[:, 1] < 0   # negative y for right
        
    points = points[indices]
    old_to_new = np.cumsum(indices) - 1
    faces = mesh.faces.reshape(-1, 4)
    valid_faces = []
    
    for face in faces[:, 1:]:
        if all(indices[face]):
            new_face = old_to_new[face]
            valid_faces.append([3] + list(new_face))
            
    valid_faces = np.array(valid_faces).flatten()
    return pv.PolyData(points, valid_faces)

def test_matlab():
    """Test reconstruction with Matlab surface"""
    print("Loading and testing Matlab surface...")
    
    # Load Matlab data
    faces = load_faces(r"C:\Users\wbou2\Documents\meg_to_surface_ml\src\data\Faces_1.mat")
    vertices = load_vertices(r"C:\Users\wbou2\Documents\meg_to_surface_ml\src\data\Vertices_1.mat")
    
    # Display original full surface
    print("\nVisualization of complete original surface:")
    pv_mesh_full = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)))
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh_full, color='lightblue')
    plotter.add_axes()
    plotter.show()
    
    # Extract left hemisphere
    print("\nExtracting left hemisphere...")
    hemisphere_mesh = extract_hemisphere(pv_mesh_full, 'left')
    
    # Display extracted hemisphere
    print("Visualizing extracted hemisphere:")
    plotter = pv.Plotter()
    plotter.add_mesh(hemisphere_mesh, color='lightblue')
    plotter.add_axes()
    plotter.show()
    
    # Prepare surface for processing
    vertices = hemisphere_mesh.points
    faces = hemisphere_mesh.faces.reshape(-1, 4)[:, 1:]
    surface_mesh = (vertices, faces)
    
    # Load template
    print("\nLoading template...")
    fsaverage6 = fetch_surf_fsaverage(mesh='fsaverage6')
    template_surf = surface.load_surf_mesh(fsaverage6['pial_left'])
    template_vertices, template_faces = template_surf[0], template_surf[1]
    
    fsav_sphere_coords, template_sphere_tris = spherical_projection_40962(
        (template_vertices, template_faces), 'lh'
    )[0:2]
    template_sphere = (fsav_sphere_coords, template_sphere_tris)
    
    # Process surface
    print("\nStarting reconstruction...")
    coords_recon, error, error_hausdorff, template_tris = process_surface(
        surface_mesh,
        template_sphere,
        lmax=10,
        lambda_reg=1e-7,
        sigma=0
    )
    
    # Get original coordinates for visualization
    sphere_coords, sphere_tris, orig_coords, orig_tris = spherical_projection_40962(
        surface_mesh, hemisphere='lh'
    )
    
    # Print results
    print(f"\nHausdorff Error: {error_hausdorff}")
    print(f"Mean Error: {np.mean(error):.4f}")
    print(f"Max Error: {np.max(error):.4f}")
    
    # Save reconstructed surface
    recon_mesh = pv.PolyData(coords_recon, 
                            np.column_stack((np.full(len(template_tris), 3), 
                                           template_tris)))
    recon_mesh.save("surface_reconstruite.vtk")
    
    # Final visualization
    visualize_results(orig_coords, coords_recon, error, template_tris)

if __name__ == "__main__":
    test_matlab()
