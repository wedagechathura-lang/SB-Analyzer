import streamlit as st
import cv2
import numpy as np
from PIL import Image
# We import cKDTree inside the function to be safe, or globally if scipy works
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# ==========================================
# 1. CORE ANALYSIS LOGIC (DOT PATTERN)
# ==========================================
def run_dot_analysis(image, modulus_mpa, poisson, strain_factor, ref_spacing_px, dot_min_area, dot_max_area):
    # --- 1. SAFETY RESIZE ---
    # Resize to max 1600px width to keep processing fast
    if image.width > 1600:
        ratio = 1600 / image.width
        new_height = int(image.height * ratio)
        image = image.resize((1600, new_height))
        
    image_np = np.array(image.convert('RGB'))
    input_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    output_image = input_bgr.copy()
    
    img_h, img_w = input_bgr.shape[:2]
    logs = []
    
    # ==========================================================
    # --- 2. TEXTURE REMOVAL & DOT DETECTION ---
    # ==========================================================
    gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    
    # Median Blur cleans fabric noise
    blurred = cv2.medianBlur(gray, 21)
    
    # Adaptive Threshold finds dots
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 15)
    
    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # --- 3. Find Dot Centroids ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dot_centers = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by dot size
        if dot_min_area < area < dot_max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dot_centers.append([cX, cY])
                
                # Draw the detected dot
                cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)

    # --- ERROR HANDLING: If no dots found ---
    if len(dot_centers) < 3:
         logs.append(f"**Error:** Only found {len(dot_centers)} dots.")
         logs.append(f"Try increasing 'Max Dot Area' (Current: {dot_max_area} px).")
         # Return the original image so the app doesn't crash
         return output_image, logs, None, binary

    logs.append(f"**Detected Dots:** {len(dot_centers)}")
    
    # ==========================================================
    # --- 4. CALCULATE STRAIN (Nearest Neighbor) ---
    # ==========================================================
    points = np.array(dot_centers)
    tree = cKDTree(points)
    
    # Find 5 nearest neighbors (1st one is the dot itself)
    distances, indices = tree.query(points, k=5) 
    
    shape_data_for_map = []
    E = modulus_mpa * 1e6 
    
    for i, (dist_list, point) in enumerate(zip(distances, points)):
        # Average distance to 4 neighbors
        local_spacing = np.mean(dist_list[1:]) 
        
        # Strain Calculation
        strain = (local_spacing - ref_spacing_px) / ref_spacing_px
        strain = strain * strain_factor
        if strain < 0: strain = 0
            
        stress_pa = E * strain
        stress_mpa = stress_pa / 1e6
        
        shape_data_for_map.append((point[0], point[1], stress_mpa))

    # ==========================================================
    # --- 5. GENERATE HEATMAP ---
    # ==========================================================
    if len(shape_data_for_map) > 0:
        try:
            h, w = output_image.shape[:2]
            # Create smaller grid for speed
            small_w, small_h = w // 10, h // 10
            small_w, small_h = max(small_w, 1), max(small_h, 1)

            map_points = np.array([(p[0]/10, p[1]/10) for p in shape_data_for_map])
            map_values = np.array([p[2] for p in shape_data_for_map])
            
            y_grid, x_grid = np.mgrid[0:small_h, 0:small_w]
            
            # Interpolate
            interpolated_map = griddata(map_points, map_values, (x_grid, y_grid), method='linear', fill_value=0)
            
            # Normalize map
            interpolated_map = np.nan_to_num(interpolated_map)
            min_s, max_s = np.min(interpolated_map), np.max(interpolated_map)
            
            if (max_s - min_s) == 0:
                normalized = np.zeros_like(interpolated_map, dtype=np.uint8)
            else:
                normalized = 255 * (interpolated_map - min_s) / (max_s - min_s)
                normalized = normalized.astype(np.uint8)

            color_map_small = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            color_map_full = cv2.resize(color_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
            
            output_image = cv2.addWeighted(color_map_full, 0.5, output_image, 0.5, 0)

        except Exception as e:
            logs.append(f"**Map Error:** {e}")

    return output_image, logs, shape_data_for_map, binary

# ==========================================
# 2. STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="Dot Pattern Analyzer", layout="wide")

st.title("🩹 Smart Bandage: Dot Pattern Mode")

# -- SIDEBAR CONTROLS --
with st.sidebar:
    st.header("1. Material Properties")
    modulus = st.number_input("Modulus (MPa)", value=51.0)
    strain_factor = st.number_input("Strain Factor", value=2.0)
    
    st.divider()
    st.header("2. Calibration")
    st.info("Adjust 'Ref Spacing' until the loose parts are Blue.")
    ref_spacing = st.slider("Ref Spacing (px)", 10.0, 100.0, 45.0, 0.5)
    
    st.divider()
    st.header("3. Dot Detection Settings")
    min_area = st.number_input("Min Dot Area (px)", value=20)
    # INCREASED DEFAULT MAX AREA to 5000 for high-res images
    max_area = st.number_input("Max Dot Area (px)", value=5000)

# -- MAIN AREA --
image_file = st.file_uploader("Upload Image (Dot Pattern)", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    if st.button("RUN FULL SURFACE ANALYSIS", type="primary"):
        with st.spinner("Analyzing Dot Matrix..."):
            original_image = Image.open(image_file)
            
            result_img, logs, map_data, binary_img = run_dot_analysis(
                original_image, modulus, 0.3, strain_factor, ref_spacing, min_area, max_area
            )
            
            # Safe Color Conversion for Streamlit
            # We convert BGR to RGB here to avoid TypeError in st.image
            if result_img is not None:
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            else:
                result_img_rgb = np.array(original_image)

            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Removed 'channels="BGR"' and used manual conversion above
                st.image(result_img_rgb, caption="Stress Heatmap", use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Analysis Logs")
                for line in logs:
                    st.markdown(line)
                
                st.divider()
                st.markdown("### 🔍 Computer Vision")
                st.caption("If this is black, increase 'Max Dot Area'")
                st.image(binary_img, caption="Detected Dots", use_container_width=True, clamp=True)
