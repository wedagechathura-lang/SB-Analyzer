import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 1. CORE ANALYSIS LOGIC
# ==========================================
def run_analysis(image, modulus_mpa, poisson, num_ovals, strain_factor, min_area_percent):
    # --- 1. SAFETY RESIZE ---
    if image.width > 1600:
        ratio = 1600 / image.width
        new_height = int(image.height * ratio)
        image = image.resize((1600, new_height))
        
    image_np = np.array(image.convert('RGB'))
    input_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    output_image = input_bgr.copy()
    
    img_h, img_w = input_bgr.shape[:2]
    total_image_pixels = img_h * img_w
    
    # Threshold Calculation
    ratio = min_area_percent / 100.0
    min_area_threshold = total_image_pixels * ratio
    
    logs = []
    logs.append(f"**Image Size:** {img_w} x {img_h} ({total_image_pixels:,} px)")
    logs.append(f"**Filter:** Ignoring circles smaller than {min_area_percent}%")
    logs.append("---")
    
    # ==========================================================
    # --- 2. IMPROVED PRE-PROCESSING (The Texture Fix) ---
    # ==========================================================
    gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    
    # STEP A: Heavy Blur to melt the fabric threads together
    # Increased from (7,7) to (35,35)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    
    # STEP B: Adaptive Threshold with a LARGER block size
    # Block size increased from 15 to 81 (must be odd number)
    # This looks at a wider area, ignoring small thread variations
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 81, 15)
    
    # STEP C: Morphological "Opening" (Erosion followed by Dilation)
    # This effectively "wipes away" any small white noise speckles left over
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # ==========================================================
    
    # --- 3. Find Contours ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_shapes = []
    
    for cnt in contours:
        # Filter very small artifacts
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 100:
            ellipse = cv2.fitEllipse(cnt)
            major = max(ellipse[1])
            minor = min(ellipse[1])
            
            if major == 0: continue
            
            ellipse_area = np.pi * (major / 2) * (minor / 2)
            
            if ellipse_area > min_area_threshold:
                aspect_ratio = minor / major
                detected_shapes.append({
                    'ellipse': ellipse, 
                    'ratio': aspect_ratio,
                    'area_px': ellipse_area,
                    'major': major,
                    'minor': minor
                })
            
    if not detected_shapes:
        return output_image, logs + [f"**Error:** No shapes found larger than {min_area_percent}%."], None, binary
        
    # --- 4. Identify Reference & Calculate Stress ---
    detected_shapes.sort(key=lambda s: s['ratio'], reverse=True)
    
    if len(detected_shapes) > num_ovals:
        detected_shapes = detected_shapes[:num_ovals]
    
    ref_shape = detected_shapes[0]
    ref_maj = ref_shape['major']
    ref_min = ref_shape['minor']
    
    E = modulus_mpa * 1e6 
    shape_data_for_map = [] 
    
    for i, s in enumerate(detected_shapes):
        center_x = int(s['ellipse'][0][0])
        center_y = int(s['ellipse'][0][1])
        center = (center_x, center_y)

        if s == ref_shape:
            cv2.ellipse(output_image, s['ellipse'], (0, 0, 255), 3) 
            logs.append(f"**Shape #{i+1} (Reference)**")
            logs.append(f"- Location: ({center_x}, {center_y})")
            logs.append(f"- Area: {s['area_px']:.0f} px")
            logs.append("---")
            cv2.putText(output_image, "Ref", (center[0] - 40, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            shape_data_for_map.append((center_x, center_y, 0.0))
        else:
            cv2.ellipse(output_image, s['ellipse'], (0, 255, 0), 3) 
            curr_maj = s['major']
            curr_min = s['minor']
            
            strain_maj = ((curr_maj - ref_maj) / ref_maj) * strain_factor
            strain_min = ((curr_min - ref_min) / ref_min) * strain_factor
            
            term = (strain_maj + poisson * strain_min)
            stress_pa = (E / (1 - poisson**2)) * term
            stress_mpa = stress_pa / 1e6
            
            logs.append(f"**Shape #{i+1} (Sensor)**")
            logs.append(f"- Location: ({center_x}, {center_y})")
            logs.append(f"- Stress: {stress_mpa:.2f} MPa")
            logs.append("---")
            
            cv2.putText(output_image, f"{stress_mpa:.2f} MPa", (center[0] - 60, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            shape_data_for_map.append((center_x, center_y, stress_mpa))

    # --- 5. TOPOGRAPHY MAP ---
    if len(shape_data_for_map) > 0:
        try:
            h, w = output_image.shape[:2]
            small_w, small_h = w // 10, h // 10
            small_w = max(small_w, 1)
            small_h = max(small_h, 1)

            points = np.array([(p[0]/10, p[1]/10) for p in shape_data_for_map]) 
            values = np.array([p[2] for p in shape_data_for_map]) 
            
            if len(points) == 1:
                interpolated_stress_map = np.full((small_h, small_w), values[0], dtype=np.float32)
            else:
                y_grid, x_grid = np.mgrid[0:small_h, 0:small_w]
                pixel_coords_flat = np.vstack([x_grid.ravel(), y_grid.ravel()]).T 
                points_reshaped = points[np.newaxis, :, :] 
                pixel_coords_reshaped = pixel_coords_flat[:, np.newaxis, :] 

                distances = np.linalg.norm(pixel_coords_reshaped - points_reshaped, axis=2)
                distances = np.maximum(distances, 1e-6) 
                
                weights = 1 / (distances**2)
                sum_of_weights = np.sum(weights, axis=1)
                interpolated_values_flat = np.sum(weights * values, axis=1) / sum_of_weights
                interpolated_stress_map = interpolated_values_flat.reshape(small_h, small_w)

            min_s, max_s = np.min(interpolated_stress_map), np.max(interpolated_stress_map)
            if (max_s - min_s) == 0:
                normalized_map = np.zeros_like(interpolated_stress_map, dtype=np.uint8)
            else:
                normalized_map = 255 * (interpolated_stress_map - min_s) / (max_s - min_s)
                normalized_map = normalized_map.astype(np.uint8)

            color_map_small = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
            color_map_full = cv2.resize(color_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
            output_image = cv2.addWeighted(color_map_full, 0.4, output_image, 0.6, 0)
            
        except Exception as e:
            logs.append(f"**Map Error:** {e}")

    return output_image, logs, shape_data_for_map, binary


# ==========================================
# 2. MOBILE APP INTERFACE
# ==========================================
st.set_page_config(page_title="Smart Bandage", layout="centered")

st.title("🩹 Smart Bandage Analyzer")
st.markdown("### Optical Pressure Monitoring System")

# -- Input Settings --
with st.expander("Analysis Parameters", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        modulus = st.number_input("Modulus (MPa)", value=51.0)
        strain_factor = st.number_input("Strain Factor", value=2.0)
        min_area_percent = st.slider(
            "Min Circle Area (% of Screen)",
            min_value=0.1, max_value=100.0, value=1.0, step=0.1, format="%.1f%%"
        )
        
    with col2:
        poisson = st.number_input("Poisson Ratio", value=0.01)
        num_ovals = st.number_input("Ovals to Find", value=2, step=1)

image_file = st.file_uploader("Take Photo or Upload", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    if st.button("RUN ANALYSIS", type="primary"):
        with st.spinner("Processing..."):
            original_image = Image.open(image_file)
            
            # UNPACK 4 ITEMS NOW
            result_img, logs, map_data, binary_img = run_analysis(
                original_image, modulus, poisson, num_ovals, strain_factor, min_area_percent
            )
            
            st.image(result_img, caption="Analysis Result", use_container_width=True, channels="BGR")
            
            st.success("Analysis Complete")
            
            # Show Logs
            for line in logs:
                st.markdown(line)
                
            # Show Binary Image (New Feature)
            st.markdown("### 🔍 Computer Vision View")
            st.caption("This is what the computer sees (Thresholded Image):")
            st.image(binary_img, caption="Binary/Threshold Mask", use_container_width=True, clamp=True)
