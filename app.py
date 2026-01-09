import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# ==========================================
# 1. CORE ANALYSIS LOGIC (DOT PATTERN)
# ==========================================
def run_dot_analysis(image, modulus_mpa, poisson, strain_factor, ref_spacing_px, dot_min_area, dot_max_area):
    # --- 1. SAFETY RESIZE ---
    # Prevents memory crashes on large images by limiting width to 1600px
    if image.width > 1600:
        ratio = 1600 / image.width
        new_height = int(image.height * ratio)
        image = image.resize((1600, new_height))
        
    image_np = np.array(image.convert('RGB'))
    # OpenCV uses BGR format internaly
    input_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    output_image = input_bgr.copy()
    
    img_h, img_w = input_bgr.shape[:2]
    
    logs = []
    
    # ==========================================================
    # --- 2. TEXTURE REMOVAL & DOT DETECTION ---
    # ==========================================================
    gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    
    # Use Median Blur to remove fabric noise but keep dot edges sharp
    blurred = cv2.medianBlur(gray, 21)
    
    # Adaptive Threshold to find dark dots on light fabric
    # Block size 51, Constant 15 tuned for fabric texture
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 15)
    
    # Clean up noise (Morphological Open to remove tiny specks)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # --- 3. Find Dot Centroids ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dot_centers = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by dot size (User adjustable from sidebar)
        if dot_min_area < area < dot_max_area:
            # Find center of mass (centroid) of the dot
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dot_centers.append([cX, cY])
                
                # Draw a green circle around detected dots on the output image
                cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 1)

    # Basic error checking if not enough dots are found
    if len(dot_centers) < 3:
         return output_image, logs + [f"**Error:** Only found {len(dot_centers)} dots. Try adjusting the Min/Max Dot Area in the sidebar."], None, binary

    logs.append(f"**Detected Dots:** {len(dot_centers)}")
    
    # ==========================================================
    # --- 4. CALCULATE STRAIN (Nearest Neighbor Distance) ---
    # ==========================================================
    # Use KDTree for efficient nearest neighbor search
    points = np.array(dot_centers)
    tree = cKDTree(points)
    
    # Find distances to the 4 nearest neighbors for every dot
    # k=5 because the query includes the point itself (distance 0)
    distances, indices = tree.query(points, k=5) 
    
    shape_data_for_map = []
    E = modulus_mpa * 1e6 # Convert MPa to Pa
    
    avg_spacings = []

    for i, (dist_list, point) in enumerate(zip(distances, points)):
        # Calculate average distance to the 4 closest neighbors (indices 1-4)
        local_spacing = np.mean(dist_list[1:]) 
        avg_spacings.append(local_spacing)
        
        # STRAIN CALCULATION
        # Formula: Strain = (Current Spacing - Reference Spacing) / Reference Spacing
        strain = (local_spacing - ref_spacing_px) / ref_spacing_px
        
        # Apply calibration factor
        strain = strain * strain_factor
        
        # Stress Calculation (Hooke's Law: Stress = E * Strain)
        # Clamp negative strain (compression) to 0 stress for visualization
        if strain < 0: strain = 0
            
        stress_pa = E * strain
        stress_mpa = stress_pa / 1e6
        
        # Store data for heatmap generation: (x, y, stress_value)
        shape_data_for_map.append((point[0], point[1], stress_mpa))

    # Calculate and log global statistics
    avg_spacing_global = np.mean(avg_spacings)
    logs.append(f"**Avg Measured Spacing:** {avg_spacing_global:.1f} px")
    logs.append(f"**Reference Spacing (Slider):** {ref_spacing_px:.1f} px")
    
    # ==========================================================
    # --- 5. GENERATE HEATMAP (Interpolation) ---
    # ==========================================================
    if len(shape_data_for_map) > 0:
        try:
            h, w = output_image.shape[:2]
            # Create a smaller grid for faster interpolation (1/10th scale) to save memory
            small_w, small_h = w // 10, h // 10
            small_w, small_h = max(small_w, 1), max(small_h, 1)

            # Prepare data for interpolation
            map_points = np.array([(p[0]/10, p[1]/10) for p in shape_data_for_map])
            map_values = np.array([p[2] for p in shape_data_for_map])
            
            # Generate the grid mesh for the new map
            y_grid, x_grid = np.mgrid[0:small_h, 0:small_w]
            
            # Perform Linear Interpolation to fill the space between dots
            interpolated_map = griddata(map_points, map_values, (x_grid, y_grid), method='linear', fill_value=0)
            
            # Clean up any NaN values at the edges resulting from interpolation
            interpolated_map = np.nan_to_num(interpolated_map)
            
            # Normalize map to 0-255 range for color mapping
            min_s, max_s = np.min(interpolated_map), np.max(interpolated_map)
            if (max_s - min_s) == 0:
                normalized = np.zeros_like(interpolated_map, dtype=np.uint8)
            else:
                normalized = 255 * (interpolated_map - min_s) / (max_s - min_s)
                normalized = normalized.astype(np.uint8)

            # Apply color map (JET = Blue to Red)
            color_map_small = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            # Resize back to full image size using linear interpolation
            color_map_full = cv2.resize(color_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Overlay the heatmap onto the original image with transparency
            output_image = cv2.addWeighted(color_map_full, 0.5, output_image, 0.5, 0)

        except Exception as e:
            logs.append(f"**Map Error:** {e}")

    # Return BGR image, logs, data, and the binary mask
    return output_image, logs, shape_data_for_map, binary

# ==========================================
# 2. STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="Dot Pattern Analyzer", layout="wide")

st.title("🩹 Smart Bandage: Dot Pattern Mode")
st.markdown("Use a full matrix of dots to detect pressure across the entire surface.")

# -- SIDEBAR CONTROLS --
with st.sidebar:
    st.header("1. Material Properties")
    modulus = st.number_input("Modulus (MPa)", value=51.0, help="Young's Modulus of the bandage material")
    strain_factor = st.number_input("Strain Factor", value=1.0, help="Calibration multiplier for strain")
    
    st.divider()
    
    st.header("2. Calibration (Crucial)")
    st.info("Adjust 'Ref Spacing' until the loose/relaxed parts of the bandage appear BLUE (0 MPa).")
    # Increased slider range to handle various image resolutions
    ref_spacing = st.slider("Ref Spacing (px)", 10.0, 200.0, 45.0, 0.5, help="The distance between dots when there is NO tension.")
    
    st.divider()
    
    st.header("3. Dot Detection Tuning")
    st.caption("Adjust these limits if dots are being missed or noise is detected as dots.")
    min_area = st.number_input("Min Dot Area (px)", value=20)
    max_area = st.number_input("Max Dot Area (px)", value=800)

# -- MAIN AREA --
image_file = st.file_uploader("Upload Image (Dot Pattern)", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    if st.button("RUN FULL SURFACE ANALYSIS", type="primary"):
        with st.spinner("Analyzing Dot Matrix..."):
            original_image = Image.open(image_file)
            
            # Run the analysis function
            result_img_bgr, logs, map_data, binary_img = run_dot_analysis(
                original_image, modulus, 0.3, strain_factor, ref_spacing, min_area, max_area
            )
            
            # --- FIX FOR TYPE ERROR ---
            # Explicitly convert BGR to RGB for Streamlit display.
            # This is more reliable than using channels="BGR".
            result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
            # ---------------------------
            
            # Create two columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display the final heatmap image using the RGB version
                st.image(result_img_rgb, caption="Stress Heatmap", use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Analysis Logs")
                for line in logs:
                    st.markdown(line)
                
                st.divider()
                st.markdown("### 🔍 Computer Vision View")
                st.caption("This is the binary mask used to find the dots.")
                # Display the thresholded binary image (clamp=True handles 0-255 range safely)
                st.image(binary_img, caption="Detected Dots (Threshold)", use_container_width=True, clamp=True)
