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
# 2. ANALYSIS LOGIC (DOUBLE-SCAN)
# ==========================================
def analyze_dot_pattern(image, modulus_mpa, poisson_ratio, strain_factor, 
                       blur_val, thresh_block, min_area, manual_spacing):
    logs = []
    
    image_np = np.array(image.convert('RGB'))
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        h, w = img.shape[:2]

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ======================================================
    # PASS 1: THE "MACRO" EYE (Find Reference Square)
    # Uses Otsu thresholding which is better for solid objects
    # ======================================================
    ref_size_px = None 
    ref_box = (0,0,0,0)
    
    # Heavy blur to merge square details
    gray_macro = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu's Threshold (Global, not Adaptive) -> Good for big solid shapes
    _, binary_macro = cv2.threshold(gray_macro, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours_macro, _ = cv2.findContours(binary_macro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort largest first
    contours_macro = sorted(contours_macro, key=cv2.contourArea, reverse=True)
    
    for cnt in contours_macro:
        area = cv2.contourArea(cnt)
        if area < 100: continue # Too small to be the square
        
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        
        # Square Check: 4 corners
        if len(approx) == 4: 
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect = float(w_rect)/h_rect
            
            # Aspect ratio check (0.7 - 1.3 allow for rotation/tilt)
            if 0.7 < aspect < 1.3:
                ref_size_px = w_rect
                ref_box = (x, y, w_rect, h_rect)
                logs.append(f"**CALIBRATION:** Found Square. 1 Unit = {ref_size_px}px")
                
                # Draw visual confirmation
                cv2.rectangle(output, (x, y), (x+w_rect, y+h_rect), (255, 0, 0), 3)
                cv2.putText(output, "REF", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                break 

    # Fallback to Manual Spacing if Square is truly gone
    if ref_size_px is None:
        if manual_spacing > 0:
            ref_size_px = manual_spacing
            logs.append(f"**CALIBRATION:** Square not found. Using Manual Spacing: {ref_size_px}px")
        else:
            return output, logs + ["ERROR: Reference Square not found. Ensure it is dark and distinct."], binary_macro

    # ======================================================
    # PASS 2: THE "MICRO" EYE (Find Dots)
    # Uses Adaptive Thresholding (User Tuned)
    # ======================================================
    if blur_val % 2 == 0: blur_val += 1
    blurred = cv2.medianBlur(gray, blur_val) 
    
    if thresh_block % 2 == 0: thresh_block += 1
    binary_micro = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, thresh_block, 2)
    
    contours_micro, _ = cv2.findContours(binary_micro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    
    # Calculate Exclusion Zone (Around the detected square)
    ex_margin = ref_size_px * 0.8
    rx, ry, rw, rh = ref_box
    safe_x1, safe_y1 = rx - ex_margin, ry - ex_margin
    safe_x2, safe_y2 = rx + rw + ex_margin, ry + rh + ex_margin
    
    for cnt in contours_micro:
        area = cv2.contourArea(cnt)
        if area < min_area: continue 
        
        # Don't re-detect the reference square contour itself
        # (It will also appear in the micro scan, but huge)
        if area > (ref_size_px * ref_size_px * 0.5): continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # --- EXCLUSION CHECK ---
            # If the dot is inside the "Reference Zone", ignore it
            if (cx > safe_x1) and (cx < safe_x2) and (cy > safe_y1) and (cy < safe_y2):
                continue
            
            dots.append([cx, cy])

    if len(dots) < 10:
        return output, logs + [f"ERROR: Found {len(dots)} dots. Adjust sliders."], binary_micro

    # --- KD-Tree Processing ---
    points = np.array(dots)
    tree = cKDTree(points)
    dist, indices = tree.query(points, k=5) 
    
    heatmap_data = [] 
    
    # Smart Spacing Logic
    all_neighbor_dists = dist[:, 1:].flatten()
    median_spacing = np.median(all_neighbor_dists)
    
    baseline_dist = ref_size_px 
    if manual_spacing > 0:
         baseline_dist = manual_spacing
         
    logs.append(f"**Baseline:** {baseline_dist:.1f}px (Median Dot Dist: {median_spacing:.1f}px)")
    
    max_valid_dist = baseline_dist * 3.0
    max_strain_val = -999.0 
    target_idx = -1
    
    border_x = w * 0.10 
    border_y = h * 0.10
    
    for i, d_list in enumerate(dist):
        neighbors = d_list[1:]
        if np.max(neighbors) > max_valid_dist: continue

        local_avg = np.mean(neighbors)
        strain = (local_avg - baseline_dist) / baseline_dist
        strain = strain * strain_factor
        strain_mag = abs(strain)
        
        px, py = points[i]
        heatmap_data.append([px, py, strain_mag])
        
        if strain > max_strain_val:
            if (px > border_x) and (px < (w - border_x)) and (py > border_y) and (py < (h - border_y)):
                max_strain_val = strain
                target_idx = i

        cv2.circle(output, (int(px), int(py)), 2, (0, 255, 0), -1)

    # --- Physics Detail ---
    sig_x_disp = 0.0
    sig_y_disp = 0.0
    hotspot_loc = (0,0)

    if target_idx != -1:
        n_indices = indices[target_idx, 1:]
        center_pt = points[target_idx]
        hotspot_loc = center_pt
        
        sx, sy = calculate_plane_stress(center_pt, points[n_indices], baseline_dist, 
                                        modulus_mpa, poisson_ratio, strain_factor)
        sig_x_disp = sx
        sig_y_disp = sy
        logs.append(f"**Peak Tension:** {sx:.2f} MPa (X)")

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
        
        # Mask Reference Area
        mask_overlay = output.copy()
        cv2.rectangle(mask_overlay, (int(safe_x1), int(safe_y1)), (int(safe_x2), int(safe_y2)), (40, 40, 40), -1)
        output = cv2.addWeighted(mask_overlay, 0.8, output, 0.2, 0)
        
        # Re-draw the Square (so it's visible on top of the mask)
        if ref_box != (0,0,0,0):
             cv2.rectangle(output, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)

        # Draw Label
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

    return output, logs, binary_micro # Return the micro mask for user to debug

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Universal Stress Analyzer", layout="wide")
st.title("🔬 Universal Stress Analyzer")

with st.sidebar:
    st.header("1. Detection Tuning")
    st.info("Adjust these if you see 'Snow' or missing dots.")
    
    blur_val = st.slider("Blur Amount (Smooths Texture)", 1, 15, 9, step=2)
    thresh_block = st.slider("Threshold Sensitivity", 3, 31, 17, step=2)
    min_area = st.slider("Min Dot Area (Pixels)", 1, 100, 5) # Default lowered to 5
    
    st.divider()
    
    st.header("2. Calibration")
    manual_spacing = st.number_input(
        "Manual Spacing Override (px)", 
        value=0.0,
        help="Use this if your dots are much smaller/closer than the Reference Square."
    )
    
    st.divider()
    
    st.header("3. Physics")
    modulus = st.number_input("Young's Modulus (MPa)", value=51.0)
    poisson = st.number_input("Poisson's Ratio", value=0.3)
    strain_factor = st.number_input("Calibration Factor", value=1.0)

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    if st.button("RUN ANALYSIS", type="primary"):
        with st.spinner("Processing..."):
            
            original_image = Image.open(image_file)
            
            result_img, logs, binary_view = analyze_dot_pattern(
                original_image, modulus, poisson, strain_factor,
                blur_val, thresh_block, min_area, manual_spacing
            )
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Stress Heatmap")
                st.image(result_img, channels="BGR", use_container_width=True)
            
            with col2:
                st.subheader("Data")
                for line in logs:
                    st.markdown(line)
                
                with st.expander("Binary Mask (Detection Check)", expanded=True):
                    st.image(binary_view, caption="White = Detected Dot", use_container_width=True)
                    st.caption("If this looks like static/snow, INCREASE 'Min Dot Area' or 'Blur'.")
