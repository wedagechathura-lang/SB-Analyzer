import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# ==========================================
# 1. PHYSICS MATH ENGINE
# ==========================================
def calculate_plane_stress(center_pt, neighbor_pts, ref_size, E, nu, factor):
    cx, cy = center_pt
    x_dists = []
    y_dists = []
    
    for nx, ny in neighbor_pts:
        dx = abs(nx - cx)
        dy = abs(ny - cy)
        dist = np.sqrt(dx**2 + dy**2)
        
        if dx > dy: x_dists.append(dist) 
        else: y_dists.append(dist) 
            
    if len(x_dists) > 0:
        avg_x = np.mean(x_dists)
        eps_x = ((avg_x - ref_size) / ref_size) * factor
    else: eps_x = 0.0
        
    if len(y_dists) > 0:
        avg_y = np.mean(y_dists)
        eps_y = ((avg_y - ref_size) / ref_size) * factor
    else: eps_y = 0.0
        
    if (1 - nu**2) == 0: K = 0 
    else: K = E / (1 - nu**2)
    
    sigma_x = K * (eps_x + (nu * eps_y))
    sigma_y = K * (eps_y + (nu * eps_x))
    
    return sigma_x, sigma_y

# ==========================================
# 2. ANALYSIS LOGIC (WITH EXCLUSION ZONE)
# ==========================================
def analyze_dot_pattern(image, modulus_mpa, poisson_ratio, strain_factor):
    logs = []
    
    image_np = np.array(image.convert('RGB'))
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Resize Logic
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- Preprocessing ---
    blurred = cv2.medianBlur(gray, 3) 
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ======================================================
    # PASS 1: FIND THE REFERENCE SQUARE ONLY
    # ======================================================
    ref_size_px = None 
    ref_box = None # (x, y, w, h)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue 
        
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        corners = len(approx)
        
        # Check for Square
        if corners == 4: 
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            ref_size_px = w_rect
            ref_box = (x, y, w_rect, h_rect)
            
            # Draw the Square
            cv2.rectangle(output, (x, y), (x+w_rect, y+h_rect), (255, 0, 0), 2)
            logs.append(f"**CALIBRATION:** Found Square at ({x},{y}). 1 Unit = {ref_size_px}px")
            break # Stop looking for the square once found

    if ref_size_px is None:
        return output, logs + ["ERROR: Reference Square not found."], binary

    # ======================================================
    # PASS 2: FIND DOTS (IGNORING THE EXCLUSION ZONE)
    # ======================================================
    dots = []
    
    # Define Exclusion Zone: 2.0x the size of the square
    # This creates a "Safe Area" around the square where we ignore everything
    exclusion_margin = ref_size_px * 2.0 
    
    ref_x, ref_y, ref_w, ref_h = ref_box
    # Calculate the forbidden rectangle coordinates
    safe_x1 = ref_x - exclusion_margin
    safe_x2 = ref_x + ref_w + exclusion_margin
    safe_y1 = ref_y - exclusion_margin
    safe_y2 = ref_y + ref_h + exclusion_margin
    
    # Draw the Exclusion Zone (Yellow Dashed Box) for visualization
    cv2.rectangle(output, (int(safe_x1), int(safe_y1)), (int(safe_x2), int(safe_y2)), (0, 255, 255), 1)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5: continue 
        
        # Skip the square itself (we already found it)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        if len(approx) == 4 and area > 100: continue

        # Get Dot Position
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # --- THE IGNORE CHECK ---
            # If the dot is inside the "Safe Box", SKIP IT.
            if (cx > safe_x1) and (cx < safe_x2) and (cy > safe_y1) and (cy < safe_y2):
                continue
            
            dots.append([cx, cy])

    if len(dots) < 10:
        return output, logs + [f"ERROR: Only found {len(dots)} valid dots."], binary

    # --- KD-Tree Processing ---
    points = np.array(dots)
    tree = cKDTree(points)
    dist, indices = tree.query(points, k=5) 
    
    heatmap_data = [] 
    
    # Dynamic Edge Filter based on Reference Size
    max_valid_dist = ref_size_px * 2.5
    
    max_strain_val = -999.0 
    target_idx = -1
    
    for i, d_list in enumerate(dist):
        neighbors = d_list[1:]
        if np.max(neighbors) > max_valid_dist: continue

        local_avg = np.mean(neighbors)
        
        # Strain Calculation
        strain = (local_avg - ref_size_px) / ref_size_px
        strain = strain * strain_factor
        
        strain_mag = abs(strain)
        heatmap_data.append([points[i][0], points[i][1], strain_mag])
        
        if strain > max_strain_val:
            max_strain_val = strain
            target_idx = i

        cv2.circle(output, (int(points[i][0]), int(points[i][1])), 2, (0, 255, 0), -1)

    # --- Physics Detail ---
    sig_x_disp = 0.0
    sig_y_disp = 0.0
    hotspot_loc = (0,0)

    if target_idx != -1:
        n_indices = indices[target_idx, 1:]
        center_pt = points[target_idx]
        hotspot_loc = center_pt
        
        sx, sy = calculate_plane_stress(center_pt, points[n_indices], ref_size_px, 
                                        modulus_mpa, poisson_ratio, strain_factor)
        sig_x_disp = sx
        sig_y_disp = sy
        
        logs.append(f"**Peak Tension:** {sx:.2f} MPa (X) / {sy:.2f} MPa (Y)")

    # --- Heatmap Generation ---
    if heatmap_data:
        h_img, w_img = img.shape[:2]
        pts = np.float32([p[:2] for p in heatmap_data])
        vals = np.float32([p[2] for p in heatmap_data])
        
        grid_x, grid_y = np.mgrid[0:w_img:4, 0:h_img:4]
        grid_z = griddata(pts, vals, (grid_x, grid_y), method='linear', fill_value=0)
        grid_z = np.nan_to_num(grid_z)
        
        peak_strain = np.percentile(vals, 99) 
        if peak_strain < 0.01: peak_strain = 0.01 
        
        norm_map = np.clip(grid_z / peak_strain, 0, 1)
        norm_map = np.power(norm_map, 0.6) 
        
        norm_uint8 = (norm_map * 255).astype('uint8').T
        full_norm = cv2.resize(norm_uint8, (w_img, h_img))
        
        color_map = cv2.applyColorMap(full_norm, cv2.COLORMAP_JET)
        
        mask = full_norm > 15 
        mask_3ch = np.dstack([mask]*3)
        output = np.where(mask_3ch, cv2.addWeighted(color_map, 0.7, output, 0.3, 0), output)
        
        if target_idx != -1:
            mx, my = int(hotspot_loc[0]), int(hotspot_loc[1])
            label_x = f"Sx: {sig_x_disp:.2f} MPa"
            label_y = f"Sy: {sig_y_disp:.2f} MPa"
            
            cv2.rectangle(output, (mx, my - 60), (mx + 220, my + 10), (255, 255, 255), -1)
            cv2.rectangle(output, (mx, my - 60), (mx + 220, my + 10), (0, 0, 0), 2)
            
            col_x = (0, 0, 255) if sig_x_disp > 0 else (255, 0, 0)
            col_y = (0, 0, 255) if sig_y_disp > 0 else (255, 0, 0)
            
            cv2.putText(output, label_x, (mx+5, my-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_x, 2)
            cv2.putText(output, label_y, (mx+5, my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_y, 2)
            cv2.circle(output, (mx, my), 5, (255, 0, 255), 2)

    # Convert to RGB for Streamlit
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    return output_rgb, logs, binary

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="High Density Analyzer", layout="wide")

st.title("🔬 High Density Stress Analyzer")

with st.sidebar:
    st.header("Physics Parameters")
    modulus = st.number_input("Young's Modulus (MPa)", value=51.0)
    poisson = st.number_input("Poisson's Ratio", value=0.3)
    strain_factor = st.number_input("Calibration Factor", value=1.0)

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    if st.button("RUN ANALYSIS", type="primary"):
        with st.spinner("Processing..."):
            
            original_image = Image.open(image_file)
            
            result_img, logs, binary_view = analyze_dot_pattern(
                original_image, modulus, poisson, strain_factor
            )
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Stress Heatmap")
                st.image(result_img, channels="BGR", use_container_width=True)
            
            with col2:
                st.subheader("Data")
                for line in logs:
                    st.markdown(line)
                
                with st.expander("Binary Mask (See what was detected)"):
                    st.image(binary_view, caption="Detected Dots", use_container_width=True)
