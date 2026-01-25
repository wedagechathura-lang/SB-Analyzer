import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# ==========================================
# 1. PHYSICS MATH ENGINE (Plane Stress)
# ==========================================
def calculate_plane_stress(center_pt, neighbor_pts, ref_size, E, nu, factor):
    """
    Separates neighbors into X (Horizontal) and Y (Vertical) to calculate
    directional stress using Poisson's Ratio.
    """
    cx, cy = center_pt
    x_dists = []
    y_dists = []
    
    for nx, ny in neighbor_pts:
        dx = abs(nx - cx)
        dy = abs(ny - cy)
        dist = np.sqrt(dx**2 + dy**2)
        
        # Classification: Is neighbor more Horizontal or Vertical?
        if dx > dy:
            x_dists.append(dist) # Horizontal Neighbor
        else:
            y_dists.append(dist) # Vertical Neighbor
            
    # Calculate Directional Strains
    if len(x_dists) > 0:
        avg_x = np.mean(x_dists)
        eps_x = ((avg_x - ref_size) / ref_size) * factor
    else:
        eps_x = 0.0
        
    if len(y_dists) > 0:
        avg_y = np.mean(y_dists)
        eps_y = ((avg_y - ref_size) / ref_size) * factor
    else:
        eps_y = 0.0
        
    # Calculate Plane Stress (Hooke's Law for 2D)
    # Factor K = E / (1 - v^2)
    if (1 - nu**2) == 0: K = 0 
    else: K = E / (1 - nu**2)
    
    sigma_x = K * (eps_x + (nu * eps_y))
    sigma_y = K * (eps_y + (nu * eps_x))
    
    return sigma_x, sigma_y

# ==========================================
# 2. ANALYSIS LOGIC (High Sensitivity)
# ==========================================
def analyze_dot_pattern(image, modulus_mpa, poisson_ratio, strain_factor):
    logs = []
    
    # --- Convert PIL to OpenCV ---
    image_np = np.array(image.convert('RGB'))
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # --- Resize if too huge ---
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        logs.append(f"Image resized to {img.shape[1]}x{img.shape[0]} for speed.")

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- Preprocessing (UPDATED FOR SMALL DOTS) ---
    # 1. Use lighter blur to preserve small dots (Was 7, now 3)
    blurred = cv2.medianBlur(gray, 3) 
    
    # 2. Use tighter thresholding (Block 11, C 2)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Optional: Dilate slightly to make tiny dots "fatter" and easier to find
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    ref_size_px = None 
    
    # --- Detection Loop ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # --- FIX: Lowered limit to 5 pixels (was 50) ---
        if area < 5: continue 
        
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        corners = len(approx)
        
        # Reference Square
        if corners == 4 and area > 500: 
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            ref_size_px = w_rect
            cv2.rectangle(output, (x, y), (x+w_rect, y+h_rect), (255, 0, 0), 3)
            logs.append(f"**CALIBRATION:** 1 Unit = {ref_size_px}px")
            
        # Dots
        else: # Accepting blobs as dots even if corners aren't perfect circles
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dots.append([cx, cy])

    if ref_size_px is None:
        return output, logs + ["ERROR: Reference Square not found."], binary
    
    if len(dots) < 5:
        return output, logs + [f"ERROR: Only found {len(dots)} dots. Check lighting."], binary

    # --- KD-Tree Processing ---
    points = np.array(dots)
    tree = cKDTree(points)
    dist, indices = tree.query(points, k=5) 
    
    heatmap_data = [] 
    max_valid_dist = ref_size_px * 1.3
    
    max_strain_val = -999.0 # Track algebraic max (Tension)
    target_idx = -1
    
    # Pass 1: Global Heatmap Data
    for i, d_list in enumerate(dist):
        neighbors = d_list[1:]
        if np.max(neighbors) > max_valid_dist: continue

        local_avg = np.mean(neighbors)
        
        # Strain Calculation
        strain = (local_avg - ref_size_px) / ref_size_px
        strain = strain * strain_factor
        
        # For Heatmap Colors: Use Magnitude
        strain_mag = abs(strain)
        heatmap_data.append([points[i][0], points[i][1], strain_mag])
        
        # For Hotspot Tracking: Look for most POSITIVE strain (Tension)
        if strain > max_strain_val:
            max_strain_val = strain
            target_idx = i

        # Draw green dot (Smaller radius for small dots)
        cv2.circle(output, (int(points[i][0]), int(points[i][1])), 2, (0, 255, 0), -1)

    # --- Pass 2: Detailed Physics for Hotspot ---
    sig_x_disp = 0.0
    sig_y_disp = 0.0
    hotspot_loc = (0,0)

    if target_idx != -1:
        n_indices = indices[target_idx, 1:]
        center_pt = points[target_idx]
        neighbor_pts = points[n_indices]
        hotspot_loc = center_pt
        
        sx, sy = calculate_plane_stress(center_pt, neighbor_pts, ref_size_px, 
                                        modulus_mpa, poisson_ratio, strain_factor)
        sig_x_disp = sx
        sig_y_disp = sy
        
        logs.append(f"**Max Tension Point:** ({center_pt[0]}, {center_pt[1]})")
        logs.append(f"- Horizontal Stress (Sx): **{sx:.2f} MPa**")
        logs.append(f"- Vertical Stress (Sy): **{sy:.2f} MPa**")

    # --- Heatmap Generation (RED = HIGH STRESS) ---
    if heatmap_data:
        h_img, w_img = img.shape[:2]
        pts = np.float32([p[:2] for p in heatmap_data])
        vals = np.float32([p[2] for p in heatmap_data])
        
        # Use finer grid for small dots
        grid_x, grid_y = np.mgrid[0:w_img:5, 0:h_img:5] # 1/5th scale for better resolution
        grid_z = griddata(pts, vals, (grid_x, grid_y), method='linear', fill_value=0)
        grid_z = np.nan_to_num(grid_z)
        
        # Dynamic Scaling
        peak_strain = np.percentile(vals, 99) 
        if peak_strain < 0.01: peak_strain = 0.01 
        
        norm_map = np.clip(grid_z / peak_strain, 0, 1)
        norm_map = np.power(norm_map, 0.6) 
        
        norm_uint8 = (norm_map * 255).astype('uint8').T
        full_norm = cv2.resize(norm_uint8, (w_img, h_img))
        
        # STANDARD JET: Low (0) = Blue, High (255) = Red
        color_map = cv2.applyColorMap(full_norm, cv2.COLORMAP_JET)
        
        # Overlay
        mask = full_norm > 15 
        mask_3ch = np.dstack([mask]*3)
        output = np.where(mask_3ch, cv2.addWeighted(color_map, 0.7, output, 0.3, 0), output)
        
        # --- DRAW DATA TAG ---
        if target_idx != -1:
            mx, my = int(hotspot_loc[0]), int(hotspot_loc[1])
            
            label_x = f"Sx: {sig_x_disp:.2f} MPa"
            label_y = f"Sy: {sig_y_disp:.2f} MPa"
            
            # Draw Box
            cv2.rectangle(output, (mx, my - 60), (mx + 220, my + 10), (255, 255, 255), -1)
            cv2.rectangle(output, (mx, my - 60), (mx + 220, my + 10), (0, 0, 0), 2)
            
            # Text Colors
            col_x = (0, 0, 255) if sig_x_disp > 0 else (255, 0, 0) # Red if +, Blue if -
            col_y = (0, 0, 255) if sig_y_disp > 0 else (255, 0, 0)
            
            cv2.putText(output, label_x, (mx + 5, my - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_x, 2)
            cv2.putText(output, label_y, (mx + 5, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_y, 2)
            
            cv2.circle(output, (mx, my), 8, (0, 255, 255), 2)
        
    return output, logs, binary

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="XY Stress Analyzer", layout="wide")

st.title("🌡️ XY Stress Analyzer")
st.markdown("### Plane Stress Analysis with Poisson Coupling")

# -- Sidebar --
with st.sidebar:
    st.header("Physics Parameters")
    modulus = st.number_input("Young's Modulus (MPa)", value=51.0, step=1.0)
    poisson = st.number_input("Poisson's Ratio", value=0.3, step=0.01)
    strain_factor = st.number_input("Calibration Factor", value=1.0, step=0.1)
    st.info("Ensure your image has the Reference Square (Red/Blue Box) visible.")

# -- Main --
image_file = st.file_uploader("Upload Dot Pattern Image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    if st.button("RUN PHYSICS ANALYSIS", type="primary"):
        with st.spinner("Calculating Stress Tensors..."):
            
            original_image = Image.open(image_file)
            
            # Run Analysis
            result_img, logs, binary_view = analyze_dot_pattern(
                original_image, modulus, poisson, strain_factor
            )
            
            # Layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Stress Heatmap")
                st.image(result_img, channels="BGR", use_container_width=True)
            
            with col2:
                st.subheader("Analysis Log")
                for line in logs:
                    st.markdown(line)
                
                with st.expander("Computer Vision View"):
                    st.image(binary_view, caption="Threshold Mask", use_container_width=True)
