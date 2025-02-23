# utils/mathutils.py
import numpy as np
from scipy import sparse
from scipy.spatial.distance import directed_hausdorff


def cart_to_sph(coords):
    x, y, z = coords.T
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # [-π, π] -> [0, 2π]
    theta = np.where(theta < 0, theta + 2*np.pi, theta)
    phi = np.arccos(np.clip(z / r, -1, 1))
    
    return r, theta, phi

def sph_to_cart(r, theta, phi):
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    
    return np.column_stack((x, y, z))

def build_template_adjacency_two_hemis(template_tris_lh, template_tris_rh):

    # Calculate total number of vertices
    vertices_lh = np.max(template_tris_lh) + 1
    vertices_rh = np.max(template_tris_rh) + 1
    total_vertices = vertices_lh + vertices_rh
    
    # Initialize vertex to faces mapping
    vertex_to_faces = [[] for _ in range(total_vertices)]
    
    # Process left hemisphere
    for face_idx, face in enumerate(template_tris_lh):
        for vertex_idx in face:
            vertex_to_faces[vertex_idx].append(face_idx)
    
    # Process right hemisphere
    n_faces_lh = len(template_tris_lh)
    for face_idx, face in enumerate(template_tris_rh):
        adjusted_face_idx = face_idx + n_faces_lh
        adjusted_vertices = face + vertices_lh  
        for vertex_idx in adjusted_vertices:
            vertex_to_faces[vertex_idx].append(adjusted_face_idx)
            
    return vertex_to_faces

def build_sparse_connectivity(faces_lh, faces_rh=None):
    if faces_rh is not None:
        vertices_lh = np.max(faces_lh) + 1
        vertices_rh = np.max(faces_rh) + 1
        total_vertices = vertices_lh + vertices_rh
        adjusted_faces_rh = faces_rh + vertices_lh
        all_faces = np.vstack([faces_lh, adjusted_faces_rh])
    else:
        total_vertices = np.max(faces_lh) + 1
        all_faces = faces_lh

    # Build rows and cols for sparse matrix
    rows, cols = [], []
    for face in all_faces:
        v1, v2, v3 = face
        rows.extend([v1, v2, v3, v2, v3, v1])
        cols.extend([v2, v3, v1, v1, v2, v3])

    data = np.ones(len(rows), dtype=bool)
    vertex_conn = sparse.csr_matrix((data, (rows, cols)), 
                                  shape=(total_vertices, total_vertices))
    
    return vertex_conn, all_faces

def get_neighbor_faces(faces, face_idx, vertex_to_faces):

    neighbor_faces = set()
    # For each vertex of the face 
    for vertex_idx in faces[face_idx]:
        # Add all the connected faces to that vertex
        neighbor_faces.update(vertex_to_faces[vertex_idx])
    return list(neighbor_faces)

def compute_face_normals(vertices, faces):

    v1 = vertices[faces[:,0]]
    v2 = vertices[faces[:,1]]
    v3 = vertices[faces[:,2]]
    
    return np.cross(v2 - v1, v3 - v1)

def compute_face_areas(vertices, faces):
    v1 = vertices[faces[:,0]]
    v2 = vertices[faces[:,1]]
    v3 = vertices[faces[:,2]]
    
    cross_products = np.cross(v2 - v1, v3 - v1)
    return 0.5 * np.sqrt(np.sum(cross_products * cross_products, axis=1))

def compute_vertex_normals(vertices, faces, vertex_to_faces, n_rings=2):

    vertex_conn, _ = build_sparse_connectivity(faces)
    face_normals = compute_face_normals(vertices, faces)
    vertex_normals = np.zeros_like(vertices)
    
    # For degenerated cases 
    n_degenerate_initial = 0
    n_fixed = 0
    n_unfixable = 0
    
    # Initial computation of the normals 
    for vertex_idx in range(len(vertices)):
        local_faces = set(vertex_to_faces[vertex_idx])
        current_ring = set(vertex_to_faces[vertex_idx])
        for _ in range(n_rings - 1):
            next_ring = set()
            for face_idx in current_ring:
                neighbors = get_neighbor_faces(faces, face_idx, vertex_to_faces)
                next_ring.update(neighbors)
            local_faces.update(next_ring)
            current_ring = next_ring
        
        local_faces = list(local_faces)
        vertex_normals[vertex_idx] = np.sum(face_normals[local_faces], axis=0)
    
    # Identify the degenerated normals 
    norms = np.linalg.norm(vertex_normals, axis=1)
    bad_normals = np.where((norms < np.finfo(float).eps) | np.isnan(norms))[0]
    n_degenerate_initial = len(bad_normals)
    
    # Correct the degenerated normals (if they exist)
    if len(bad_normals) > 0:
        for bad_idx in bad_normals:
            neighbors = vertex_conn[bad_idx].indices
            # Eclude the nehibors with degenerated faces 
            valid_neighbors = np.setdiff1d(neighbors, bad_normals)
            
            if len(valid_neighbors) > 0:
                # Mean of valid normals 
                new_normal = np.mean(vertex_normals[valid_neighbors], axis=0)
                new_norm = np.linalg.norm(new_normal)
                if new_norm > np.finfo(float).eps:
                    vertex_normals[bad_idx] = new_normal / new_norm
                    n_fixed += 1
                else:
                    vertex_normals[bad_idx] = np.array([1., 0., 0.])
                    n_unfixable += 1
            else:
                vertex_normals[bad_idx] = np.array([1., 0., 0.])
                n_unfixable += 1
    
    # Normalization 
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return vertex_normals / norms

def compute_mean_curvature(vertices, faces, vertex_to_faces):

    vertex_conn, _ = build_sparse_connectivity(faces)
    
    #Diagonal matrices of the coordinates
    Dx = sparse.diags(vertices[:,0])
    Dy = sparse.diags(vertices[:,1])
    Dz = sparse.diags(vertices[:,2])
    
    # Computation of the edges
    Ex = vertex_conn @ Dx - Dx @ vertex_conn
    Ey = vertex_conn @ Dy - Dy @ vertex_conn
    Ez = vertex_conn @ Dz - Dz @ vertex_conn
    
    # Normalization of the edges 
    En = np.sqrt(Ex.multiply(Ex) + Ey.multiply(Ey) + Ez.multiply(Ez))
    mask = En.data > 0
    En.data[mask] = 1.0 / En.data[mask]
    
    Ex = Ex.multiply(En)
    Ey = Ey.multiply(En)
    Ez = Ez.multiply(En)
    
    # Normals of the vertices 
    vertex_normals = compute_vertex_normals(vertices, faces, vertex_to_faces)
    
    # Scalar product of the normals 
    Nx = sparse.diags(vertex_normals[:,0])
    Ny = sparse.diags(vertex_normals[:,1])
    Nz = sparse.diags(vertex_normals[:,2])
    
    Ip = (Nx @ Ex + Ny @ Ey + Nz @ Ez).tocsr()
    
    # Angles and mean curvature (mean of the angles )
    angles = np.arccos(np.clip(Ip.data, -1, 1))
    curvature = np.zeros(len(vertices))
    
    for i in range(len(vertices)):
        row_start = Ip.indptr[i]
        row_end = Ip.indptr[i+1]
        if row_end > row_start:
            curvature[i] = np.mean(angles[row_start:row_end]) - np.pi/2
            
    return curvature

def hausdorff_distance(array1, array2):
    distance_1 = directed_hausdorff(array1, array2)[0]
    distance_2 = directed_hausdorff(array2, array1)[0]
    return max(distance_1, distance_2)

def compute_normal_differences(normals1, normals2):

    # Scalar product between normals
    dots = np.sum(normals1 * normals2, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angle in degrees
    angles = np.degrees(np.arccos(dots))
    
    return {
        'mean_angle': np.mean(angles),
        'std_angle': np.std(angles),
        'max_angle': np.max(angles),
        'percentile_95': np.percentile(angles, 95),
        'num_large_errors': np.sum(angles > 90),
        'percent_large_errors': 100 * np.sum(angles > 80) / len(angles),
        'angle_distribution': angles
    }

def compute_curvature_differences(curv1, curv2):
    abs_diff = np.abs(curv1 - curv2)
    
    return {
        'mean_abs_diff': np.mean(abs_diff),
        'std_abs_diff': np.std(abs_diff),
        'max_abs_diff': np.max(abs_diff),
        'percentile_95': np.percentile(abs_diff, 95),
        'percentile_75': np.percentile(abs_diff, 75),
        'original_stats': {
            'mean': np.mean(curv1),
            'std': np.std(curv1),
            'min': np.min(curv1),
            'max': np.max(curv1)
        },
        'compared_stats': {
            'mean': np.mean(curv2),
            'std': np.std(curv2),
            'min': np.min(curv2),
            'max': np.max(curv2)
        },
        'difference_distribution': abs_diff
    }
def compute_point_distances(points1, points2):

    distances = np.sqrt(np.sum((points1 - points2)**2, axis=1))
    char_size = np.max(np.ptp(points1, axis=0))  
    
    return {
        'mean_dist': np.mean(distances),
        'std_dist': np.std(distances),
        'max_dist': np.max(distances),
        'normalized_mean': np.mean(distances) / char_size,
        'normalized_max': np.max(distances) / char_size,
        'percentile_95': np.percentile(distances, 95),
        'characteristic_size': char_size,
        'distance_distribution': distances
    }

def compute_hausdorff_metrics(points1, points2):

    forward_dist = directed_hausdorff(points1, points2)[0]
    backward_dist = directed_hausdorff(points2, points1)[0]
    char_size = np.max(np.ptp(points1, axis=0))
    
    return {
        'hausdorff_dist': max(forward_dist, backward_dist),
        'forward_dist': forward_dist,
        'backward_dist': backward_dist,
        'normalized_hausdorff': max(forward_dist, backward_dist) / char_size,
        'characteristic_size': char_size
    }