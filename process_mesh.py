import trimesh
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt

# --- New Libraries for Bonus Task ---
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

# --- Configuration ---
# List of meshes to run through Tasks 1-3
MESH_FILES_TO_PROCESS = ['person.obj', 'table.obj', 'talwar.obj']
# The single mesh to use for the advanced Bonus Task
BONUS_TASK_MESH = 'person.obj'

NUM_BINS = 1024 # Bin size for UNIFORM quantization

# Bonus Task Config
MIN_BINS_ADAPTIVE = 256
MAX_BINS_ADAPTIVE = 2048

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input_meshes')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Function ---

def save_and_visualize_mesh(mesh, filename, window_name, show_visuals=True):
    """Saves a mesh to the output folder and optionally visualizes it."""
    
    # Save the mesh
    output_path = os.path.join(OUTPUT_DIR, filename)
    mesh.export(output_path)
    # Don't print save messages for every file, only for key ones
    if show_visuals:
        print(f"Saved mesh to: {output_path}")
    
    # Visualize the mesh
    if show_visuals:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        
        print(f"Visualizing: {window_name}. Close the window to continue...")
        o3d.visualization.draw_geometries([o3d_mesh], window_name=window_name)

# --- Task 1 ---

def task1_load_and_inspect(mesh_file, show_visuals=True):
    """
    Loads a mesh, prints its statistics, and optionally visualizes it.
    """
    print(f"\n--- Task 1: Load and Inspect: {mesh_file} ---")
    
    mesh_path = os.path.join(INPUT_DIR, mesh_file)
    try:
        mesh = trimesh.load(mesh_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None, None

    vertices = mesh.vertices
    
    if not isinstance(vertices, np.ndarray) or vertices.size == 0:
        print("Mesh loaded, but it contains no vertices.")
        return None, None

    print(f"[Statistics - {mesh_file}]")
    print(f"Total Vertices: {len(vertices)}")
    
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_mean = vertices.mean(axis=0)
    
    print(f"Min (X,Y,Z): {v_min}")
    print(f"Max (X,Y,Z): {v_max}")
    print(f"Mean (X,Y,Z): {v_mean}")
    
    # Visualize the original mesh (optional)
    if show_visuals:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()
        
        print("\nVisualizing original mesh... Close the window to continue.")
        o3d.visualization.draw_geometries([o3d_mesh], window_name=f"Original: {mesh_file}")
    
    print("--- Task 1 Complete ---")
    return mesh, (v_min, v_max, v_mean)

# --- Task 2 ---

def task2_normalize_and_quantize(original_mesh, mesh_stats, mesh_filename, show_visuals=True):
    """
    Applies two normalization methods and quantizes the mesh.
    """
    if show_visuals:
        print(f"\n--- Task 2: Normalize and Quantize ({mesh_filename}) ---")
    
    vertices = original_mesh.vertices
    faces = original_mesh.faces
    v_min, v_max, v_mean = mesh_stats
    
    quantization_data = {}

    # --- Method 1: Min-Max Normalization ---
    v_range = v_max - v_min
    v_range[v_range == 0] = 1e-6 
    norm_minmax = (vertices - v_min) / v_range
    quant_minmax = np.floor(norm_minmax * (NUM_BINS - 1)).astype(int)
    
    mesh_quant_minmax = trimesh.Trimesh(vertices=quant_minmax, faces=faces)
    
    save_and_visualize_mesh(
        mesh_quant_minmax,
        f"{mesh_filename}_quant_minmax.obj",
        "Normalized (Min-Max)",
        show_visuals=show_visuals
    )
    
    quantization_data['min_max'] = {
        'quantized': quant_minmax,
        'stats': (v_min, v_range) 
    }

    # --- Method 2: Unit Sphere Normalization ---
    centered_vertices = vertices - v_mean
    max_distance = np.max(np.linalg.norm(centered_vertices, axis=1))
    norm_unitsphere = centered_vertices / max_distance
    norm_unitsphere_scaled = (norm_unitsphere + 1) / 2.0
    
    quant_unitsphere = np.floor(norm_unitsphere_scaled * (NUM_BINS - 1)).astype(int)
    
    mesh_quant_unitsphere = trimesh.Trimesh(vertices=quant_unitsphere, faces=faces)

    save_and_visualize_mesh(
        mesh_quant_unitsphere,
        f"{mesh_filename}_quant_unitsphere.obj",
        "Normalized (Unit Sphere)",
        show_visuals=show_visuals
    )
    
    quantization_data['unit_sphere'] = {
        'quantized': quant_unitsphere,
        'stats': (v_mean, max_distance) 
    }

    if show_visuals:
        print("\n--- Task 2 Complete ---")
    return quantization_data

# --- Task 3 ---

def task3_reconstruct_and_measure_error(original_mesh, quant_data, mesh_filename, show_visuals=True):
    """
    Dequantizes, denormalizes, and measures error.
    """
    if show_visuals:
        print(f"\n--- Task 3: Dequantize, Denormalize, and Measure Error ({mesh_filename}) ---")
    
    original_vertices = original_mesh.vertices
    faces = original_mesh.faces
    error_metrics = {}

    # --- Method 1: Min-Max Reconstruction ---
    q_minmax = quant_data['min_max']['quantized']
    dequant_minmax = q_minmax / (NUM_BINS - 1)
    v_min, v_range = quant_data['min_max']['stats']
    recon_minmax = dequant_minmax * v_range + v_min
    
    mse_minmax = np.mean((original_vertices - recon_minmax) ** 2, axis=0)
    total_mse_minmax = np.mean(mse_minmax)
    error_metrics['Min-Max'] = mse_minmax
    
    if show_visuals:
        print(f"Reconstructed MSE (Min-Max): Total:  {total_mse_minmax:.8f}")

    mesh_recon_minmax = trimesh.Trimesh(vertices=recon_minmax, faces=faces)
    save_and_visualize_mesh(
        mesh_recon_minmax,
        f"{mesh_filename}_recon_minmax.obj",
        "Reconstructed (Min-Max)",
        show_visuals=show_visuals
    )

    # --- Method 2: Unit Sphere Reconstruction ---
    q_unitsphere = quant_data['unit_sphere']['quantized']
    dequant_unitsphere = q_unitsphere / (NUM_BINS - 1)
    dequant_unitsphere_scaled = dequant_unitsphere * 2.0 - 1.0
    v_mean, max_distance = quant_data['unit_sphere']['stats']
    recon_unitsphere = dequant_unitsphere_scaled * max_distance + v_mean
    
    mse_unitsphere = np.mean((original_vertices - recon_unitsphere) ** 2, axis=0)
    total_mse_unitsphere = np.mean(mse_unitsphere)
    error_metrics['Unit Sphere'] = mse_unitsphere

    if show_visuals:
        print(f"Reconstructed MSE (Unit Sphere): Total:  {total_mse_unitsphere:.8f}")

    mesh_recon_unitsphere = trimesh.Trimesh(vertices=recon_unitsphere, faces=faces)
    save_and_visualize_mesh(
        mesh_recon_unitsphere,
        f"{mesh_filename}_recon_unitsphere.obj",
        "Reconstructed (Unit Sphere)",
        show_visuals=show_visuals
    )

    # --- 5. Plot reconstruction error ---
    labels = ['X-axis', 'Y-axis', 'Z-axis']
    minmax_errors = error_metrics['Min-Max']
    unitsphere_errors = error_metrics['Unit Sphere']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width/2, minmax_errors, width, label='Min-Max')
    ax.bar(x + width/2, unitsphere_errors, width, label='Unit Sphere')

    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(f'Reconstruction Error per Axis ({mesh_filename})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(OUTPUT_DIR, f"{mesh_filename}_error_plot.png")
    plt.savefig(plot_path)
    plt.close(fig) # Close the figure to save memory
    print(f"Saved error plot to: {plot_path}")

    if show_visuals:
        print("\n--- Task 3 Complete ---")
    return error_metrics


# --- Bonus Task ---

def task_bonus_adaptive_quantization(original_mesh, mesh_filename):
    """
    Implements normalization and adaptive quantization for a single mesh.
    """
    print(f"\n--- Bonus Task: Invariance and Adaptive Quantization ({mesh_filename}) ---")
    
    original_vertices = original_mesh.vertices
    faces = original_mesh.faces
    
    # --- 1. Generate Rotated/Translated Version ---
    angle = np.pi / 4 # 45 degrees
    rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    trans_vector = np.array([10, -5, 20])
    transform = rot_matrix.copy()
    transform[:3, 3] = trans_vector
    
    transformed_mesh = original_mesh.copy()
    transformed_mesh.apply_transform(transform)
    
    print(f"Generated a transformed copy (rotated 45 deg, translated).")

    # --- 2. Implement Invariant Normalization (PCA) ---
    def invariant_normalize(mesh):
        vertices = mesh.vertices
        v_mean = vertices.mean(axis=0)
        centered_vertices = vertices - v_mean
        
        pca = PCA(n_components=3)
        pca.fit(centered_vertices)
        
        aligned_vertices = pca.transform(centered_vertices)
        
        # We find the min/max of the *aligned* vertices
        # This creates a tight bounding box, which is better than a sphere
        v_min_aligned = aligned_vertices.min(axis=0)
        v_max_aligned = aligned_vertices.max(axis=0)
        v_range_aligned = v_max_aligned - v_min_aligned
        v_range_aligned[v_range_aligned == 0] = 1e-6
        
        # Scale to [0, 1]
        norm_scaled = (aligned_vertices - v_min_aligned) / v_range_aligned
        
        return norm_scaled, v_mean, pca.components_, v_min_aligned, v_range_aligned

    norm_orig, v_mean_orig, pca_comp_orig, v_min_orig, v_range_orig = invariant_normalize(original_mesh)
    norm_trans, _, _, _, _ = invariant_normalize(transformed_mesh)

    invariance_error = np.mean((norm_orig - norm_trans) ** 2)
    print(f"\n[Normalization Invariance Check]")
    print(f"MSE between normalized original and transformed: {invariance_error:.10f}")
    if invariance_error < 1e-6:
        print("SUCCESS: PCA-based normalization is invariant.")
    
    # --- 3. Analyze Local Density ---
    print("\n[Adaptive Quantization]")
    print("Analyzing local vertex density...")
    
    tree = KDTree(norm_orig)
    distances, _ = tree.query(norm_orig, k=11) 
    mean_neighbor_dist = np.mean(distances[:, 1:], axis=1)
    
    min_dist, max_dist = mean_neighbor_dist.min(), mean_neighbor_dist.max()
    # Invert: high density (low dist) -> 1.0, low density (high dist) -> 0.0
    density_metric = (max_dist - mean_neighbor_dist) / (max_dist - min_dist)
    
    # --- 4. Quantize with Variable Bin Sizes ---
    adaptive_bins = MIN_BINS_ADAPTIVE + (density_metric * (MAX_BINS_ADAPTIVE - MIN_BINS_ADAPTIVE))
    adaptive_bins = adaptive_bins.reshape(-1, 1) # Shape (n_vertices, 1)
    
    quant_adaptive = np.floor(norm_orig * (adaptive_bins - 1)).astype(int)
    print("Quantized mesh using adaptive bins.")

    # --- 5. Dequantize and Denormalize ---
    dequant_adaptive = quant_adaptive / (adaptive_bins - 1)
    
    # Denormalize (reverse steps from invariant_normalize)
    dequant_aligned = dequant_adaptive * v_range_orig + v_min_orig
    recon_centered = dequant_aligned.dot(pca_comp_orig)
    recon_adaptive = recon_centered + v_mean_orig
    print("Dequantized and denormalized mesh.")

    # --- 6. Measure and Plot Error ---
    mse_adaptive = np.mean((original_vertices - recon_adaptive) ** 2, axis=0)
    print(f"\nReconstructed MSE (PCA + Adaptive): Total: {np.mean(mse_adaptive):.8f}")

    # Visualize the reconstructed mesh
    mesh_recon_adaptive = trimesh.Trimesh(vertices=recon_adaptive, faces=faces)
    save_and_visualize_mesh(
        mesh_recon_adaptive,
        f"{mesh_filename}_recon_adaptive.obj",
        "Reconstructed (PCA + Adaptive)",
        show_visuals=True
    )
    
    print("\n--- Bonus Task Complete ---")
    return mse_adaptive

# --- Plotting Function ---
def plot_final_comparison(all_errors, mesh_filename):
    """Plots a final graph comparing all methods for one mesh."""
    print(f"\n--- Final Results and Analysis ({mesh_filename}) ---")
    
    labels = ['X-axis', 'Y-axis', 'Z-axis']
    num_methods = len(all_errors)
    
    if num_methods == 0:
        print("No error data to plot.")
        return

    x = np.arange(len(labels))
    width = 0.8 / num_methods
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, (method_name, errors) in enumerate(all_errors.items()):
        offset = (i - (num_methods - 1) / 2) * width
        ax.bar(x + offset, errors, width, label=f"{method_name} (Total MSE: {np.mean(errors):.8f})")

    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(f'Final Reconstruction Error Comparison ({mesh_filename})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f"{mesh_filename}_FINAL_COMPARISON_PLOT.png")
    plt.savefig(plot_path)
    print(f"Saved final comparison plot to: {plot_path}")
    
    print("\nVisualizing final comparison plot... Close the window to finish.")
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    
    # --- Run Tasks 1-3 for all specified meshes ---
    print("--- Processing All Meshes (Tasks 1-3) ---")
    all_mesh_errors = {}
    
    for mesh_file in MESH_FILES_TO_PROCESS:
        task1_results = task1_load_and_inspect(mesh_file, show_visuals=False)
        
        if task1_results[0]:
            original_mesh, mesh_stats = task1_results
            
            quant_data = task2_normalize_and_quantize(
                original_mesh, mesh_stats, mesh_file, show_visuals=False
            )
            
            if quant_data:
                errors = task3_reconstruct_and_measure_error(
                    original_mesh, quant_data, mesh_file, show_visuals=False
                )
                all_mesh_errors[mesh_file] = errors
        else:
            print(f"Skipping {mesh_file} due to loading error.")
    
    print("\n--- Main Task Batch Processing Complete ---")
    print(f"Error plots for all {len(MESH_FILES_TO_PROCESS)} meshes are saved in the 'output' folder.")

    # --- Run Bonus Task for the single specified mesh ---
    print(f"\n\n--- Starting Bonus Task Deep Dive ({BONUS_TASK_MESH}) ---")
    
    # We need to re-load the mesh and run the tasks again, this time with visuals
    task1_results = task1_load_and_inspect(BONUS_TASK_MESH, show_visuals=True)
    
    if task1_results[0]:
        original_mesh, mesh_stats = task1_results
        
        quant_data = task2_normalize_and_quantize(
            original_mesh, mesh_stats, BONUS_TASK_MESH, show_visuals=True
        )
        
        task3_errors = task3_reconstruct_and_measure_error(
            original_mesh, quant_data, BONUS_TASK_MESH, show_visuals=True
        )
        
        # Run the actual Bonus Task
        try:
            bonus_errors = task_bonus_adaptive_quantization(original_mesh, BONUS_TASK_MESH)
            
            # Combine all errors for the final plot
            final_errors_for_plot = {
                'Min-Max': task3_errors['Min-Max'],
                'Unit Sphere': task3_errors['Unit Sphere'],
                'PCA + Adaptive': bonus_errors
            }
            
            # Plot the final comparison
            plot_final_comparison(final_errors_for_plot, BONUS_TASK_MESH)
            
        except Exception as e:
            print(f"\n--- Error running Bonus Task ---")
            print(f"Error: {e}")
            print("Please ensure 'scikit-learn' and 'scipy' are installed.")

    print("\n--- All Tasks Complete ---")