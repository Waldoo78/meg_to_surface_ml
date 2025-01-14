import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import sparse
import pyvista as pv
import os
import pickle
import matplotlib.pyplot as plt

# Utils import
from utils.file_manip.vtk_processing import convert_triangles_to_pyvista
from utils.cortical import spherical_harmonics as SH
from utils.mathutils import cart_to_sph

# Load data paths and templates
data_path = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\src\cortical_transformation\data"
main_folder = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\Anatomy_data_CAM_CAN"
template_projection_lh = np.load(os.path.join(data_path, "lh_sphere_projection.npz"))
template_projection_rh = np.load(os.path.join(data_path, "rh_sphere_projection.npz"))

# Load subject coefficients
coeffs_all_lh = {}
coeffs_all_rh = {}
for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)
    if os.path.isdir(folder_path):
        with open(os.path.join(folder_path, "coeffs_lh.pkl"), 'rb') as f:
            coeffs_lh = pickle.load(f)
        coeffs_all_lh[folder] = coeffs_lh["organized_coeffs"]
        with open(os.path.join(folder_path, "coeffs_rh.pkl"), 'rb') as f:
            coeffs_rh = pickle.load(f)
        coeffs_all_rh[folder] = coeffs_rh["organized_coeffs"]

# Load fsaverage data
fsaverage_path = r"C:\Users\wbou2\Desktop\meg_to_surface_ml\data\fsaverage"
with open(os.path.join(fsaverage_path, "coeffs_lh.pkl"), 'rb') as f:
    coeffs_fsav_lh = pickle.load(f)['organized_coeffs']
fsav_data_lh = np.load(os.path.join(fsaverage_path, "lh_resampled.npz"))
coords_fsav_lh = fsav_data_lh['coords']
tris_fsav_lh = fsav_data_lh['tris']

with open(os.path.join(fsaverage_path, "coeffs_rh.pkl"), 'rb') as f:
    coeffs_fsav_rh = pickle.load(f)['organized_coeffs']
fsav_data_rh = np.load(os.path.join(fsaverage_path, "rh_resampled.npz"))
coords_fsav_rh = fsav_data_rh['coords']
tris_fsav_rh = fsav_data_rh['tris']

# Load harmonics
Y_lh_full = np.load(os.path.join(data_path, "Y_lh.npz"))['Y']
Y_rh_full = np.load(os.path.join(data_path, "Y_rh.npz"))['Y']
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pyvista as pv
from utils.mathutils import (compute_vertex_normals, compute_mean_curvature, 
                           compute_curvature_differences, compute_hausdorff_metrics, 
                           compute_point_distances, compute_normal_differences, 
                           build_template_adjacency_two_hemis)
from utils.cortical import spherical_harmonics as SH
from utils.file_manip.vtk_processing import convert_triangles_to_pyvista
import matplotlib.pyplot as plt

class ComplexAutoencoder(nn.Module):
    def __init__(self, input_size, reduction_factor):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = max(2, input_size // reduction_factor)
        self.encoder = nn.Sequential(
            nn.Linear(input_size * 2, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, input_size * 2)
        )
        
    def forward(self, x):
        x_combined = torch.cat([torch.real(x), torch.imag(x)], dim=-1)
        latent = self.encoder(x_combined)
        output = self.decoder(latent)
        n = output.shape[-1] // 2
        return torch.complex(output[..., :n], output[..., n:])

def complex_loss(output, target):
    # Split real and imaginary parts for phase comparison
    output_real = torch.real(output)
    output_imag = torch.imag(output)
    target_real = torch.real(target)
    target_imag = torch.imag(target)
    
    # Amplitude loss
    amp_loss = F.mse_loss(torch.abs(output), torch.abs(target))
    
    # Phase loss using real and imaginary components separately
    phase_loss_real = F.mse_loss(output_real / (torch.abs(output) + 1e-8), 
                                target_real / (torch.abs(target) + 1e-8))
    phase_loss_imag = F.mse_loss(output_imag / (torch.abs(output) + 1e-8), 
                                target_imag / (torch.abs(target) + 1e-8))
    phase_loss = (phase_loss_real + phase_loss_imag) / 2
    
    return 0.5 * amp_loss + 0.5 * phase_loss

def train_autoencoder_for_order(l, coeffs_diff, device='cuda', reduction_factor=3, validation_split=0.2):
    n_subjects = len(coeffs_diff)
    data = {comp: np.zeros((n_subjects, 2*l+1), dtype=complex) for comp in ['x', 'y', 'z']}
    
    for i, subject in enumerate(coeffs_diff.keys()):
        for m in range(-l, l+1):
            m_idx = m + l
            for j, comp in enumerate(['x', 'y', 'z']):
                data[comp][i, m_idx] = coeffs_diff[subject][l][m][j]
    
    data = {comp: torch.tensor(arr, dtype=torch.cfloat).to(device) 
            for comp, arr in data.items()}
    
    # Create validation split
    dataset = TensorDataset(*[data[comp] for comp in ['x', 'y', 'z']])
    val_size = int(n_subjects * validation_split)
    train_size = n_subjects - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=min(8, train_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_size)
    
    input_size = 2*l+1
    autoencoders = {comp: ComplexAutoencoder(input_size, reduction_factor).to(device) 
                    for comp in ['x', 'y', 'z']}
    optimizers = {comp: optim.AdamW(ae.parameters(), lr=0.001, weight_decay=1e-5) 
                 for comp, ae in autoencoders.items()}
    schedulers = {comp: optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=20) 
                 for comp, opt in optimizers.items()}
    
    best_val_loss = float('inf')
    best_models = {}
    patience = 50
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(2000):
        # Training
        train_loss = 0
        for ae in autoencoders.values():
            ae.train()
            
        for batch_x, batch_y, batch_z in train_loader:
            for comp, batch in zip(['x', 'y', 'z'], [batch_x, batch_y, batch_z]):
                optimizers[comp].zero_grad()
                output = autoencoders[comp](batch)
                loss = complex_loss(output, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(autoencoders[comp].parameters(), 1.0)
                optimizers[comp].step()
                train_loss += loss.item()
        
        # Validation
        val_loss = 0
        for ae in autoencoders.values():
            ae.eval()
            
        with torch.no_grad():
            for val_x, val_y, val_z in val_loader:
                for comp, batch in zip(['x', 'y', 'z'], [val_x, val_y, val_z]):
                    output = autoencoders[comp](batch)
                    val_loss += complex_loss(output, batch).item()
        
        # Update schedulers
        for comp in autoencoders:
            schedulers[comp].step(val_loss)
        
        # Track losses
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_models = {comp: ae.state_dict().copy() 
                         for comp, ae in autoencoders.items()}
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 100 == 0:
            print(f'l={l}, Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.6f}, '
                  f'Val Loss: {val_loss/len(val_loader):.6f}')
    
    # Load best models
    for comp in autoencoders:
        autoencoders[comp].load_state_dict(best_models[comp])
        
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves for l={l}')
    plt.legend()
    plt.grid(True)
    plt.close()
    
    return tuple(autoencoders[comp] for comp in ['x', 'y', 'z'])

def evaluate_reconstruction(surface_orig, surface_ae, faces, vertex_to_faces):
    """Evaluate reconstruction quality using multiple metrics"""
    # Point distances
    point_metrics = compute_point_distances(surface_orig, surface_ae)
    
    # Normal differences
    normals_orig = compute_vertex_normals(surface_orig, faces, vertex_to_faces)
    normals_ae = compute_vertex_normals(surface_ae, faces, vertex_to_faces)
    normal_metrics = compute_normal_differences(normals_orig, normals_ae)
    
    # Curvature differences
    curv_orig = compute_mean_curvature(surface_orig, faces, vertex_to_faces)
    curv_ae = compute_mean_curvature(surface_ae, faces, vertex_to_faces)
    curvature_metrics = compute_curvature_differences(curv_orig, curv_ae)
    
    # Hausdorff distance
    hausdorff_metrics = compute_hausdorff_metrics(surface_orig, surface_ae)
    
    return {
        'point_metrics': point_metrics,
        'normal_metrics': normal_metrics,
        'curvature_metrics': curvature_metrics,
        'hausdorff_metrics': hausdorff_metrics
    }

def compute_average_metrics(metrics_list):
    """Compute average metrics across all orders"""
    if not metrics_list:
        return None
        
    avg_metrics = {
        'point': {
            'mean_dist': np.mean([m['point_metrics']['mean_dist'] for m in metrics_list]),
            'max_dist': np.mean([m['point_metrics']['max_dist'] for m in metrics_list]),
            'normalized_mean': np.mean([m['point_metrics']['normalized_mean'] for m in metrics_list])
        },
        'normal': {
            'mean_angle': np.mean([m['normal_metrics']['mean_angle'] for m in metrics_list]),
            'percent_large_errors': np.mean([m['normal_metrics']['percent_large_errors'] for m in metrics_list])
        },
        'curvature': {
            'mean_abs_diff': np.mean([m['curvature_metrics']['mean_abs_diff'] for m in metrics_list]),
            'max_abs_diff': np.mean([m['curvature_metrics']['max_abs_diff'] for m in metrics_list])
        },
        'hausdorff': {
            'hausdorff_dist': np.mean([m['hausdorff_metrics']['hausdorff_dist'] for m in metrics_list]),
            'normalized_hausdorff': np.mean([m['hausdorff_metrics']['normalized_hausdorff'] for m in metrics_list])
        }
    }
    return avg_metrics

def generate_surface_diff_AE(Y, subject_id, coeffs_diff, fsav_coeffs, lmax, surface_orig, sigma=0):
   xyz_total = SH.generate_surface(Y, lmax, sigma, fsav_coeffs).astype(np.complex128)
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
   train_coeffs = coeffs_diff.copy()
   del train_coeffs[subject_id]
   
   for l in range(1, lmax + 1):
       if l <= 10:
           for idx, comp in enumerate(['x', 'y', 'z']):
               Y_block = Y[:, l*l:l*l + 2*l+1]
               coeffs = [coeffs_diff[subject_id][l][m][idx] for m in range(-l, l+1)]
               xyz_total[:, idx] += np.exp(-l*(l+1)*sigma) * (Y_block @ coeffs)
           continue
           
       ae_x, ae_y, ae_z = train_autoencoder_for_order(l, train_coeffs, device)
       
       for comp, ae in zip(['x', 'y', 'z'], [ae_x, ae_y, ae_z]):
           idx = {'x': 0, 'y': 1, 'z': 2}[comp]
           diff = np.array([coeffs_diff[subject_id][l][m][idx] for m in range(-l, l+1)])
           
           with torch.no_grad():
               diff_tensor = torch.tensor(diff, dtype=torch.cfloat).to(device)
               decoded = ae(diff_tensor).cpu().numpy()
               
               Y_block = Y[:, l*l:l*l + 2*l+1]
               xyz_total[:, idx] += np.exp(-l*(l+1)*sigma) * (Y_block @ decoded)
   
   xyz_real = np.real(xyz_total)
   xyz_centered = xyz_real - np.mean(xyz_real, axis=0)
   scale_factor = np.std(surface_orig) / np.std(xyz_centered)
   return xyz_centered * scale_factor



def visualize_reconstruction(surface_orig, surface_ae, template_faces):
    p = pv.Plotter(shape=(1, 2))
    tris_pv = convert_triangles_to_pyvista(template_faces)
    
    p.subplot(0, 0)
    mesh_orig = pv.PolyData(surface_orig, tris_pv)
    p.add_mesh(mesh_orig, show_edges=True, color='white')
    p.add_text("Original Surface", position='upper_edge')
    
    p.subplot(0, 1)
    mesh_ae = pv.PolyData(surface_ae, tris_pv)
    p.add_mesh(mesh_ae, show_edges=True, color='white')
    p.add_text("Reconstructed Surface", position='upper_edge')
    
    p.link_views()
    p.view_isometric()
    p.show(auto_close=True)

if __name__ == "__main__":
   lmax = 40
   sigma = 1e-7
   sub_name = "sub-CC110033"
   
   surface_lh_orig = SH.generate_surface(Y_lh_full, lmax=lmax, sigma=0, orders=coeffs_all_lh[sub_name])
   surface_lh_ae = generate_surface_diff_AE(
       Y=Y_lh_full,
       subject_id=sub_name,
       coeffs_diff=coeffs_all_lh,
       fsav_coeffs=coeffs_fsav_lh,
       surface_orig=surface_lh_orig,
       lmax=lmax,
       sigma=sigma
   )
   
   visualize_reconstruction(surface_lh_orig, surface_lh_ae, tris_fsav_lh)