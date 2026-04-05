import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN # NEW: for robust outlier removal

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
# 2. ANALYSIS LOGIC
# ==========================================
def analyze_dot_pattern(image, modulus_mpa, poisson_ratio, strain_factor, 
                       blur_val, thresh_block, min_area, manual_spacing, ref_adj_perc):
    logs = []
    
    image_np = np.array(image.convert('RGB'))
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    # Limit max size to prevent crashes
    scale = 1.0
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        h, w = img.shape[:2]

    output = img.copy()
    
    # ======================================================
    # PASS 1: FIND RED REFERENCE SQUARE (COLOR DETECTION)
    # ======================================================
    detected_ref_px = None 
    ref_box = (0,0,0,0)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
    
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_red:
        largest_red = max(contours_red, key=cv2.contourArea)
        if cv2.contourArea(largest_red) > 100:
            x, y, w_rect, h_rect = cv2.boundingRect(largest_red)
            detected_ref_px = w_rect
            ref_box = (x, y, w_rect, h_rect)
            logs.append(f"**DETECTION:** Found Red Square: {detected_ref_px}px")
            cv2.rectangle(output, (x, y), (x+w_rect, y+h_rect), (0, 255, 0), 3) 
    
    # Fallback: Black Square Shape
    if detected_ref_px is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_macro = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_macro = cv2.threshold(gray_macro, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours_macro, _ = cv2.findContours(binary_macro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_macro:
            area = cv2.contourArea(cnt)
            if area < 100: continue
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                if 0.7 < float(w_rect)/h_rect < 1.3:
                    detected_ref_px = w_rect
                    ref_box = (x, y, w_rect, h_rect)
                    logs.append(f"**DETECTION:** Found Black Square: {detected_ref_px}px")
                    break

    # Determine Base Reference Value
    base_ref_px = detected_ref_px
    if detected_ref_px is None:
        if manual_spacing > 0:
            base_ref_px = manual_spacing
            logs.append(f"**CALIBRATION:** Using Manual Spacing Overide: {base_ref_px}px")
        else:
            return output, logs + ["ERROR: Reference Square not found. Check lighting or use Manual Spacing."], cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img
    elif manual_spacing > 0:
         base_ref_px = manual_spacing
         logs.append(f"**CALIBRATION:** Overriding detected square with Manual Spacing: {base_ref_px}px")

    # MODIFICATION: APPLY SLIDER ADJUSTMENT
    baseline_dist = base_ref_px * (1.0 + (ref_adj_perc / 100.0))
    logs.append(f"**CALIBRATION:** Final Baseline Reference = {baseline_dist:.2f}px")

    # ======================================================
    # PASS 2: FIND DOTS (REWRITTEN FOR ROBUSTNESS)
    # ======================================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Illumination Correction (Existing, essential for shadows)
    kernel_dim = max(9, int(h * 0.02))
    if kernel_dim % 2 == 0: kernel_dim += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_dim, kernel_dim))
    corrected_dots_bright = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Standard Filtering (Keep user's high Blur: 11)
    if blur_val % 2 == 0: blur_val += 1
    blurred = cv2.medianBlur(corrected_dots_bright, blur_val) 

    # Binary Segmentation (Keep user's preferred high Sensitivity T: 3)
    # This binary image will look VERY noisy/grainy in the background. Good.
    if thresh_block % 2 == 0: thresh_block += 1
    binary_micro = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, thresh_block, 2)
    
    # Verification Image setup
    verification_img = img.copy()

    contours_micro, _ = cv2.findContours(binary_micro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # visualize contours considered candidates (before area filtering) in blue
    cv2.drawContours(verification_img, contours_micro, -1, (255, 100, 0), 1)

    dots = []
    # Use base detected size for masking, not adjusted size
    mask_ref_size = base_ref_px if base_ref_px is not None else manual_spacing
    ex_margin = mask_ref_size * 0.8
    rx, ry, rw, rh = ref_box
    safe_x1, safe_y1 = rx - ex_margin, ry - ex_margin
    safe_x2, safe_y2 = rx + rw + ex_margin, ry + rh + ex_margin
    
    for cnt in contours_micro:
        area = cv2.contourArea(cnt)
        if area < min_area: continue 
        # Use adjusted baseline for max dot size filtering logic
        if area > (baseline_dist * baseline_dist * 0.5): continue

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Masking logic
            if ref_box != (0,0,0,0):
                if (cx > safe_x1) and (cx < safe_x2) and (cy > safe_y1) and (cy < safe_y2):
                    continue
            
            dots.append([cx, cy])

    if len(dots) < 10:
        return output, logs + [f"ERROR: Only {len(dots)} valid candidate blobs found. Adaptive thresholding window may be breaking locally. (Check Verification Expander)"], binary_micro, verification_img

    # ==========================================
    # MODIFICATION PHASES START HERE
    # ==========================================
    raw_points = np.array(dots)
    
    # --- New Phase 1: Robust Outlier Removal (DBSCAN Clustering) ---
    # Fiber noise usually results in isolated clusters or long thin noise artifacts. 
    # Real dots are densely packed in a cluster with standardized spatial consensus.
    # set Eps to roughly 2x the expected baseline to allow for stretch, but reject far noise.
    db = DBSCAN(eps=baseline_dist * 2.0, min_samples=4).fit(raw_points)
    labels = db.labels_
    
    # Find the main dense cluster (picked by largest count)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    validated_dots = []
    inpainted_dots = []

    if len(unique_labels) == 0:
        return output, logs + ["ERROR: Failed to find a coherent pattern cluster. Adaptive threshold likely created a noisy mess. Re-tune parameters."], binary_micro, verification_img
        
    main_cluster_label = unique_labels[np.argmax(counts)]
    
    # Extract points belonging to the main cluster and visualize outliers in red cross
    clean_points_list = []
    for i, p in enumerate(raw_points):
        px, py = int(p[0]), int(p[1])
        if labels[i] == main_cluster_label:
            clean_points_list.append(p)
            validated_dots.append(p)
            # Visualization modification: Draw green circles on correctly detected AND pattern-passed dots
            cv2.circle(verification_img, (px, py), 3, (0, 255, 0), -1)
        elif labels[i] == -1:
            # Visualize outliers officially rejected by pattern logic with red cross
            cv2.drawMarker(verification_img, (px, py), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    logs.append(f"**PATTERN CLEANING:** Retained {len(clean_points_list)} blobs (DBSCAN picked cluster with {np.max(counts)} points).")
    
    final_points = np.array(clean_points_list)

    # --- New Phase 2: Missing Dot Imputation (KD-Tree Inpainting) ---
    # Hooke's Law requires neighbors. Boundary points or holes break the physics engine local expectation.
    # Since it is a printed grid, we can deduce "holes" in the pattern based on neighborhood statistical properties.
    pattern_tree = cKDTree(final_points)
    dist, indices = pattern_tree.query(final_points, k=5) 
    
    all_neighbor_dists = dist[:, 1:].flatten()
    median_spacing = np.median(all_neighbor_dists)
    
    logs.append(f"**Median Dot Distance:** {median_spacing:.1f}px (embedded physics input)")

    # Identify points with $<4$ immediate grid neighbors and synthesize the missing neighbor.
    imputed_points = []
    
    # Analyze the geometry of the core cluster to find holes
    for i, d_list in enumerate(dist):
        neighbors_idx = indices[i, 1:]
        current_pt = final_points[i]
        
        # Immediate neighborhood geometry check (simplified)
        missing_neighbor = False
        potential_neighbor_v = np.array([0,0])
        
        # Check if neighbors are clustered only on one side (meaning it's a boundary point with a hole nearby)
        neighbor_vs = final_points[neighbors_idx] - current_pt
        mean_v = np.mean(neighbor_vs, axis=0)
        
        # If clustered geometry found (meaning boundary/hole)
        if np.linalg.norm(mean_v) > median_spacing * 0.6:
            # We found a gap. Missing neighbor is likely on the opposite side of the clustered mean_v.
            potential_neighbor_v = -mean_v
            missing_neighbor = True
            
        if missing_neighbor:
            # We found a gap. Validate if the hole is real by checking if ANY final_point exists there.
            # Ideal hole location: current + (ideal pitch vector). We use mean_v as proxy.
            ideal_hole_loc = current_pt + potential_neighbor_v
            
            # Use kdtree to check if ANY point exists near this ideal hole
            hole_dist, _ = pattern_tree.query(ideal_hole_loc, k=1)
            
            # Tolerance for hole matching
            if hole_dist > median_spacing * 1.5: # 1.5 means no dot exists roughly where one *should* be
                 # Found a hole! Synthesize the missing neighbor
                 new_p = ideal_hole_loc.astype(int)
                 imputed_points.append(new_p)
                 
    logs.append(f"**INPAINTING:** Synthesized {len(imputed_points)} artificial dots to fill pattern holes.")
    
    # VISUALIZE IMPUTED DOTS IN YELLOW on validation image
    for p in imputed_points:
        px, py = int(p[0]), int(p[1])
        cv2.circle(verification_img, (px, py), 4, (0, 255, 255), -1)

    # Blend validated and imputed dots for final physics processing
    final_processing_list = validated_dots + imputed_points
    points = np.array(final_processing_list)
    
    # Re-run KD-Tree on complete, cleaned dataset for physical model
    tree = cKDTree(points)
    dist, indices = tree.query(points, k=5) 
    
    # heatmap data loop continues as normal using this cleaned list.
    heatmap_data = [] 
    max_valid_dist = baseline_dist * 3.0
    
    for i, d_list in enumerate(dist):
        neighbors = d_list[1:]
        if np.max(neighbors) > max_valid_dist: continue

        local_avg = np.mean(neighbors)
        strain = (local_avg - baseline_dist) / baseline_dist
        strain = strain * strain_factor
        px, py = points[i]
        heatmap_data.append([px, py, strain])
        
    # ==========================================
    # FIND IMAGE MID POINT STRESS
    # ==========================================
    sig_x_disp = 0.0
    sig_y_disp = 0.0
    img_center = np.array([w / 2, h / 2])
    
    center_dist, target_idx = tree.query(img_center, k=1)
    
    hotspot_loc = points[target_idx]
    n_indices = indices[target_idx, 1:]
    center_pt = points[target_idx]
    
    sx, sy = calculate_plane_stress(center_pt, points[n_indices], baseline_dist, 
                                    modulus_mpa, poisson_ratio, strain_factor)
    sig_x_disp = sx
    sig_y_disp = sy
    
    logs.append(f"**Analysis Point:** Closest blob to center at {hotspot_loc}")
    logs.append(f"**Center Stress:** {sx:.2f} MPa (X)")

    # --- Heatmap Generation (CONTINUOUS JET) ---
    if heatmap_data:
        h_img, w_img = img.shape[:2]
        pts = np.float32([p[:2] for p in heatmap_data])
        vals = np.float32([p[2] for p in heatmap_data])
        
        grid_x, grid_y = np.mgrid[0:w_img:4, 0:h_img:4]
        grid_z = griddata(pts, vals, (grid_x, grid_y), method='linear', fill_value=0)
        grid_z = np.nan_to_num(grid_z)
        
        full_grid_z = cv2.resize(grid_z.T, (w_img, h_img)) 
        
        limit = max(abs(np.min(full_grid_z)), abs(np.max(full_grid_z)))
        if limit < 0.01: limit = 0.01
        
        norm_map = (full_grid_z + limit) / (2 * limit)
        norm_map = np.clip(norm_map, 0, 1)
        full_norm_uint8 = (norm_map * 255).astype('uint8')
        
        color_map = cv2.applyColorMap(full_norm_uint8, cv2.COLORMAP_JET)
        
        hull_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        hull_pts = cv2.convexHull(points.astype(np.int32)) # use complete dataset for hull mask
        cv2.fillPoly(hull_mask, [hull_pts], 255)
        
        mask_3ch = np.dstack([hull_mask]*3) > 128
        
        output = np.where(mask_3ch, cv2.addWeighted(color_map, 0.6, output, 0.4, 0), output)
        
        # Mask Reference Area visual
        if ref_box != (0,0,0,0):
             cv2.rectangle(output, (int(safe_x1), int(safe_y1)), (int(safe_x2), int(safe_y2)), (40, 40, 40), -1)
             cv2.rectangle(output, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)

        # Draw Label for Mid Point stress
        mx, my = int(hotspot_loc[0]), int(hotspot_loc[1])
        label_x = f"Sx: {sig_x_disp:.2f} MPa"
        label_y = f"Sy: {sig_y_disp:.2f} MPa"
        
        header = "Center Stress"
        
        cv2.rectangle(output, (mx, my - 75), (mx + 220, my + 10), (255, 255, 255), -1)
        cv2.rectangle(output, (mx, my - 75), (mx + 220, my + 10), (0, 0, 0), 2)
        
        col_x = (0, 0, 255) if sig_x_disp > 0 else (255, 0, 0)
        col_y = (0, 0, 255) if sig_y_disp > 0 else (255, 0, 0)
        
        cv2.putText(output, header, (mx+5, my-55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(output, label_x, (mx+5, my-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_x, 2, cv2.LINE_AA)
        cv2.putText(output, label_y, (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_y, 2, cv2.LINE_AA)
        
        cv2.circle(output, (mx, my), 5, (255, 0, 255), 2, cv2.LINE_AA)

    return output, logs, binary_micro, verification_img 

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Universal Stress Analyzer", layout="wide")
st.title("🔬 Universal Stress Analyzer")

with st.sidebar:
    st.header("1. Detection Tuning")
    st.info("Adjust these if you see extreme noise or missing blobs. Compare Verification Map.")
    
    blur_val = st.slider("Noise Reduction Blur", 1, 15, 11, step=2, help="User optimized Blur:11 recommended.")
    thresh_block = st.slider("Threshold Sensitivity", 3, 31, 3, step=2, help="User optimized T:3 (High Sensitivity) recommended.")
    min_area = st.slider("Min Dot Area (Pixels)", 1, 100, 5, help="Filters out noise chunks/fibers based on bounded pixel size.")
    
    st.divider()
    st.header("2. Calibration")
    manual_spacing = st.number_input("Manual Spacing Override (px)", value=0.0, help="Manually set baseline pixel distance between dots, ignoring detected square.")
    
    ref_adj_perc = st.slider(
        "Fine-tune Reference Baseline (%)", 
        min_value=-20.0, 
        max_value=20.0, 
        value=0.0, 
        step=0.1,
        help="Adjust the calculated resting dot distance. Positive increases resting length (shifting map towards compression), negative decreases it. Use to set specific map regions to zero stress."
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
            result_img, logs, binary_view, dots_view = analyze_dot_pattern(
                original_image, modulus, poisson, strain_factor,
                blur_val, thresh_block, min_area, manual_spacing,
                ref_adj_perc
            )
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Stress Topography heat map")
                st.image(result_img, channels="BGR", use_container_width=True)
            
            with col2:
                st.subheader("Analysis Data")
                for line in logs: st.markdown(line)
                
                # --- VERIFICATION SECTION ---
                st.divider()
                st.subheader("Computer Vision Verification")
                st.info("Observe if noise is being incorrectly marked as dots. GREEN marks core grid blobs, YELLOW marks inpainted blobs.")
                
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    with st.expander("1. Raw Candidate Mask", expanded=True):
                        # adaptiveThreshold output (grayscale)
                        st.image(binary_view, caption="Shows all noise (grain/fibers)", use_container_width=True)
                with exp_col2:
                    with st.expander("2. Valid Blobs Detected", expanded=True):
                        # BGR image with overlays
                        st.image(dots_view, caption="G: Valid, Y: Artificial, Blue: Candidates, Red: Outliers", use_container_width=True, channels="BGR")
