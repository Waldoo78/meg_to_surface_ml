import numpy as np
import mne
from mne.io.constants import FIFF
import scipy.io as sio
from scipy import signal, linalg
import warnings


def compute_leadfield(meg_channel_path, lh_vertices, lh_faces, rh_vertices, rh_faces):
    """
    Compute MEG leadfield from channel information and cortical surface.
    
    Parameters
    ----------
    meg_channel_path : str
        Path to the channel_vectorview306 .mat file
    lh_vertices, rh_vertices : array-like
        Vertices coordinates for left and right hemispheres
    lh_faces, rh_faces : array-like
        Faces indices for left and right hemispheres
        
    Returns
    -------
    leadfield : np.ndarray
        The computed leadfield matrix
    fwd : dict
        The complete forward solution from MNE
    transform_matrix : np.ndarray
        Transformation matrix
    """
    
    print(f"Loading MEG channel info from: {meg_channel_path}")
    
    # Load MEG channel info
    ch_info = sio.loadmat(meg_channel_path)
    channels = ch_info['Channel'][0][:306]
    head_points = ch_info['HeadPoints'][0][0]
    
    # Build channel information
    ch_names, ch_types, ch_locs = [], [], []
    for i, ch in enumerate(channels):
        ch_type = ch['Type'][0]
        if ch_type not in ['MEG GRAD', 'MEG MAG']:
            continue
            
        # Handle channel name
        try:
            name = ch['Name'][0][0]
            if name in ['S', 'M']:
                ch_names.append(f"{name}{i+1:03d}")
            else:
                ch_names.append(name)
        except:
            ch_names.append(f"MEG{i+1:03d}")
            
        ch_types.append('grad' if ch_type == 'MEG GRAD' else 'mag')
        
        # Handle channel location
        loc = np.zeros(12)
        try:
            if ch['Loc'].size > 0:
                loc[:3] = ch['Loc'][:, 0]
                if ch_type == 'MEG GRAD' and ch['Loc'].shape[1] > 1:
                    loc[3:6] = ch['Loc'][:, 1]
                    if ch['Orient'].size > 0:
                        loc[6:9] = ch['Orient'][:, 0]
                        loc[9:12] = ch['Orient'][:, 1]
                else:
                    loc[3:6] = loc[:3]
                    if ch['Orient'].size > 0:
                        loc[6:9] = ch['Orient'][:, 0]
                        loc[9:12] = loc[6:9]
        except:
            pass  # Keep zeros if loading fails
            
        ch_locs.append(loc)
    
    print(f"Found {len(ch_names)} valid MEG channels")
    
    # Create MNE info structure
    info = mne.create_info(ch_names, sfreq=1000, ch_types=ch_types, verbose=False)
    
    # Update channel information
    for i, (name, ch_type, loc) in enumerate(zip(ch_names, ch_types, ch_locs)):
        coil_type = FIFF.FIFFV_COIL_VV_PLANAR_T1 if ch_type == 'grad' else FIFF.FIFFV_COIL_VV_MAG_T1
        
        info['chs'][i].update({
            'ch_name': name,
            'kind': FIFF.FIFFV_MEG_CH,
            'coil_type': coil_type,
            'loc': loc,
            'coord_frame': FIFF.FIFFV_COORD_HEAD,
            'unit': FIFF.FIFF_UNIT_T if ch_type == 'mag' else FIFF.FIFF_UNIT_T_M,
            'cal': 1.0,
            'range': 1.0
        })
    
    # Setup digitization points
    dig = []
    point_type_to_kind = {
        'LPA': FIFF.FIFFV_POINT_LPA,
        'NAS': FIFF.FIFFV_POINT_NASION,
        'RPA': FIFF.FIFFV_POINT_RPA,
        'HPI': FIFF.FIFFV_POINT_HPI,
        'EXTRA': FIFF.FIFFV_POINT_EXTRA
    }
    
    try:
        for i in range(head_points[0].shape[1]):
            try:
                point_type = str(head_points[2][0][i][0][0])
                kind = point_type_to_kind.get(point_type, FIFF.FIFFV_POINT_EXTRA)
                dig.append({
                    'kind': kind,
                    'ident': i,
                    'r': head_points[0][:, i],
                    'coord_frame': FIFF.FIFFV_COORD_HEAD
                })
            except:
                continue
    except:
        print("Warning: Could not process head points")
    
    # Create and apply montage
    nasion = next((d['r'] for d in dig if d['kind'] == FIFF.FIFFV_POINT_NASION), None)
    lpa = next((d['r'] for d in dig if d['kind'] == FIFF.FIFFV_POINT_LPA), None)
    rpa = next((d['r'] for d in dig if d['kind'] == FIFF.FIFFV_POINT_RPA), None)
    hsp = [d['r'] for d in dig if d['kind'] == FIFF.FIFFV_POINT_EXTRA]
    
    if nasion is not None and lpa is not None and rpa is not None:
        montage = mne.channels.make_dig_montage(
            nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp, coord_frame='head'
        )
        info.set_montage(montage, verbose=False)
    
    # Set device->head transform
    info['dev_head_t'] = mne.transforms.Transform('meg', 'head', np.eye(4))
    
    # Clean projections
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_temp = mne.io.RawArray(np.zeros((len(ch_names), 1000)), info, verbose=False)
        raw_temp.del_proj()
        info = raw_temp.info
    
    # Transform vertices to head coordinate system
    transform_matrix = ch_info['TransfMeg'][0][2]
    def transform_points(points):
        points_homog = np.hstack([points, np.ones((points.shape[0], 1))])
        return (points_homog @ transform_matrix.T)[:, :3]
    
    lh_vertices_t = transform_points(lh_vertices)
    rh_vertices_t = transform_points(rh_vertices)
    
    print(f"Transformed vertices - LH: {lh_vertices_t.shape}, RH: {rh_vertices_t.shape}")
    
    # Compute vertex normals
    def build_vertex_to_faces(faces):
        v2f = [[] for _ in range(np.max(faces) + 1)]
        for idx, face in enumerate(faces):
            for vertex in face:
                v2f[vertex].append(idx)
        return v2f
    
    def compute_normals(vertices, faces, v2f):
        normals = np.zeros_like(vertices)
        for i, vertex_faces in enumerate(v2f):
            if not vertex_faces:
                continue
            face_normals = []
            for face_idx in vertex_faces:
                face = faces[face_idx]
                v0, v1, v2 = vertices[face]
                normal = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    face_normals.append(normal / norm)
            if face_normals:
                normal = np.mean(face_normals, axis=0)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normals[i] = normal / norm
        return normals
    
    # Build source space
    v2f_lh = build_vertex_to_faces(lh_faces)
    v2f_rh = build_vertex_to_faces(rh_faces)
    
    lh_normals = compute_normals(lh_vertices_t, lh_faces, v2f_lh)
    rh_normals = compute_normals(rh_vertices_t, rh_faces, v2f_rh)
    
    src = mne.SourceSpaces([
        dict(
            rr=lh_vertices_t,
            nn=lh_normals,
            tris=lh_faces,
            np=len(lh_vertices_t),
            ntri=len(lh_faces),
            type='surf',
            id=101,
            coord_frame=FIFF.FIFFV_COORD_HEAD,
            nuse=len(lh_vertices_t),
            inuse=np.ones(len(lh_vertices_t), dtype=int),
            vertno=np.arange(len(lh_vertices_t)),
            use_tris=np.arange(len(lh_faces)),
            nuse_tri=len(lh_faces),
            nearest=None,
            nearest_dist=None,
            pinfo=None,
            patch_inds=None,
            dist=None,
            dist_limit=None
        ),
        dict(
            rr=rh_vertices_t,
            nn=rh_normals,
            tris=rh_faces,
            np=len(rh_vertices_t),
            ntri=len(rh_faces),
            type='surf',
            id=102,
            coord_frame=FIFF.FIFFV_COORD_HEAD,
            nuse=len(rh_vertices_t),
            inuse=np.ones(len(rh_vertices_t), dtype=int),
            vertno=np.arange(len(rh_vertices_t)),
            use_tris=np.arange(len(rh_faces)),
            nuse_tri=len(rh_faces),
            nearest=None,
            nearest_dist=None,
            pinfo=None,
            patch_inds=None,
            dist=None,
            dist_limit=None
        )
    ])
    
    # Compute forward solution with version compatibility
    print("Computing forward solution...")
    sphere = mne.make_sphere_model(head_radius="auto", info=info, verbose=False)
    
    # Try force_fixed first (newer MNE), then fixed (older MNE)
    try:
        fwd = mne.make_forward_solution(
            info=info, src=src, bem=sphere, trans=None,
            meg=True, eeg=False, force_fixed=True, surf_ori=True, verbose=False
        )
    except TypeError:
        fwd = mne.make_forward_solution(
            info=info, src=src, bem=sphere, trans=None,
            meg=True, eeg=False, fixed=True, surf_ori=True, verbose=False
        )
    
    leadfield = fwd['sol']['data']
    print(f"Leadfield computed - Shape: {leadfield.shape}")
    
    return leadfield, fwd, transform_matrix


def temporal_reduction(Y_extract, sfreq=1000, freq_range=(8, 30), variance_explained=0.9):
    """
    Temporal reduction of a MEG/EEG signal extract using DCT basis.
    
    Parameters
    ----------
    Y_extract : np.ndarray
        Multi-channel signal (n_channels, n_times)
    sfreq : float
        Sampling frequency (Hz)
    freq_range : tuple
        Frequency range to preserve (Hz)
    variance_explained : float
        Percentage of variance to preserve (0-1)
    
    Returns
    -------
    Y_reduced : np.ndarray
        Signal after temporal reduction
    P : np.ndarray
        Projection matrix
    var_cum : np.ndarray
        Cumulative explained variance
    n_modes : int
        Number of modes used
    """
    
    n_channels, n_times = Y_extract.shape
    print(f"Temporal reduction: {Y_extract.shape} -> freq range {freq_range} Hz")
    
    # Apply Hanning window
    w = signal.windows.hann(n_times, sym=False)
    Y_windowed = Y_extract * w[np.newaxis, :]
    
    # Compute covariance matrix
    W_tilde = Y_windowed.T @ Y_windowed
    
    # Calculate indices for DCT in the frequency range
    k_min = int(np.ceil(freq_range[0] * 2 * n_times / sfreq))
    k_max = int(np.floor(freq_range[1] * 2 * n_times / sfreq))
    k_min = max(0, k_min)
    k_max = min(n_times - 1, k_max)
    
    if k_min > k_max:
        raise ValueError(f"Frequency range {freq_range} too narrow for {n_times} time points")
    
    k = np.arange(k_min, k_max + 1)
    print(f"Using DCT modes {k_min}-{k_max} ({len(k)} modes)")
    
    # Construct restricted DCT basis
    t = np.arange(n_times)[:, np.newaxis]
    K = np.sqrt(2.0 / n_times) * np.cos(np.pi * k * (2 * t + 1) / (2 * n_times))
    
    # Project covariance into DCT space and eigendecomposition
    K_tilde = K.T @ W_tilde @ K
    K_tilde = (K_tilde + K_tilde.T) / 2  # Ensure symmetry
    
    S, U = linalg.eigh(K_tilde)
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    
    # Select modes to achieve variance_explained
    S_positive = S[S > 0]
    var_cum = np.cumsum(S_positive) / np.sum(S_positive)
    n_modes = np.searchsorted(var_cum, variance_explained) + 1
    n_modes = min(n_modes, len(S_positive))
    
    U = U[:, :n_modes]
    print(f"Selected {n_modes} modes explaining {var_cum[n_modes-1]:.1%} of variance")
    
    # Construct projection matrix P (size: n_times x n_modes)
    PW = w[:, np.newaxis] * K
    P = PW @ U
    P = P / np.linalg.norm(P, axis=0, keepdims=True)  # Normalize
    
    # Project the data
    Y_reduced = Y_windowed @ P
    
    return Y_reduced, P, var_cum[:n_modes], n_modes


def check_covariance_properties(Qe, name="Matrix"):
    """Check and display key properties of a covariance matrix."""
    
    # Calculate all properties
    is_symmetric = np.allclose(Qe, Qe.T, rtol=1e-10)
    eigenvals = np.linalg.eigvals(Qe)
    eigenvals = np.real(eigenvals)
    
    is_positive_definite = np.all(eigenvals > 1e-12)
    has_positive_diagonal = np.all(np.diag(Qe) > 0)
    cond = np.linalg.cond(Qe)
    rank = np.linalg.matrix_rank(Qe)
    
    # Display results
    print(f"{name} properties ({Qe.shape[0]}×{Qe.shape[1]}):")
    print(f"  - Symmetric: {is_symmetric}")
    print(f"  - Positive definite: {is_positive_definite}")
    print(f"  - Positive diagonal: {has_positive_diagonal}")
    print(f"  - Rank: {rank}/{Qe.shape[0]}")
    print(f"  - Condition number: {cond:.2e}")
    print(f"  - Eigenvalue range: [{np.min(eigenvals):.3e}, {np.max(eigenvals):.3e}]")
    
    if cond > 1e12:
        print(f"  ⚠ Warning: Matrix is ill-conditioned")
    
    return {
        'symmetric': is_symmetric,
        'positive_definite': is_positive_definite,
        'condition_number': cond,
        'eigenvalues': eigenvals
    }


def regularize_covariance(Qe, percentile=50):
    """
    Regularize a covariance matrix using eigenvalue thresholding.
    
    Parameters
    ----------
    Qe : np.ndarray
        Input covariance matrix
    percentile : float
        Percentile for eigenvalue flooring (0-100)
        
    Returns
    -------
    Qe_reg : np.ndarray
        Regularized covariance matrix
    """
    
    print(f"Regularizing matrix with percentile={percentile}")
    
    # Diagonalize and threshold
    eigenvals, eigenvecs = np.linalg.eigh(Qe)
    threshold = np.percentile(eigenvals, percentile)
    eigenvals_reg = np.maximum(eigenvals, threshold)
    
    n_modified = np.sum(eigenvals < threshold)
    print(f"Modified {n_modified}/{len(eigenvals)} eigenvalues (threshold: {threshold:.3e})")
    
    # Reconstruct matrix ensuring symmetry
    Qe_reg = eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T
    Qe_reg = (Qe_reg + Qe_reg.T) / 2
    
    print(f"Condition number: {np.linalg.cond(Qe):.2e} -> {np.linalg.cond(Qe_reg):.2e}")
    
    return Qe_reg