import streamlit as st
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

# ==========================================
# 1. PHYSICS & ANALYSIS ENGINE
# ==========================================
class StressEngine:
    def __init__(self, e_modulus, poisson_ratio, strain_factor):
        self.E = e_modulus
        self.nu = poisson_ratio
        self.factor = strain_factor

    def get_plane_stress(self, center_pt, neighbors, baseline_dist):
        """Calculates Sigma X and Y based on dot displacement."""
        cx, cy = center_pt
        x_dists = [np.linalg.norm(n - center_pt) for n in neighbors if abs(n[0] - cx) > abs(n[1] - cy)]
        y_dists = [np.linalg.norm(n - center_pt) for n in neighbors if abs(n[1] - cy) > abs(n[0] - cx)]

        # Calculate Strain: (Current - Baseline) / Baseline
        eps_x = ((np.mean(x_dists) - baseline_dist) / baseline_dist) * self.factor if x_dists else 0
        eps_y = ((np.mean(y_dists) - baseline_dist) / baseline_dist) * self.factor if y_dists else 0

        # Plane Stress Transformation
        k = self.E / (1 - self.nu**2)
        sigma_x = k * (eps_x + (self.nu * eps_y))
        sigma_y = k * (eps_y + (self.nu * eps_x))
        
        return sigma_x, sigma_y

class ImageProcessor:
    @staticmethod
    def find_reference_square(img_hsv, img_bgr):
        """Detects the red calibration square to determine pixel scaling."""
        # Red has two ranges in HSV
        m1 = cv2.inRange(img_hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        m2 = cv2.inRange(img_hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(m1 + m2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                return cv2.boundingRect(largest)
        return None

    @staticmethod
    def get_dots(gray, blur, block_size, min_area):
        """Extracts dot coordinates from the speckle pattern."""
        blurred = cv2.medianBlur(gray, blur)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, 2
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    dots.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
        return np.array(dots), binary

# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def run_analysis(image, params):
    logs = []
    # 1. Setup Images
    img_bgr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    
    proc = ImageProcessor()
    engine = StressEngine(params['modulus'], params['poisson'], params['factor'])

    # 2. Reference Calibration
    ref_box = proc.find_reference_square(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), img_bgr)
    base_px = ref_box[2] if ref_box else params['manual_spacing']
    
    if base_px <= 0:
        return None, ["ERROR: No reference found. Set Manual Spacing."], None, None

    # Apply fine-tuning adjustment
    baseline_dist = base_px * (1.0 + (params['adj_perc'] / 100.0))
    logs.append(f"📏 **Baseline:** {baseline_dist:.2f}px")

    # 3. Dot Detection
    dots, binary_mask = proc.get_dots(gray, params['blur'], params['thresh'], params['min_area'])
    
    # Filter dots near reference square
    if ref_box:
        rx, ry, rw, rh = ref_box
        margin = rw * 0.8
        mask = ~((dots[:,0] > rx-margin) & (dots[:,0] < rx+rw+margin) & 
                 (dots[:,1] > ry-margin) & (dots[:,1] < ry+rh+margin))
        dots = dots[mask]

    if len(dots) < 10:
        return None, ["ERROR: Too few dots detected."], binary_mask, img_bgr

    # 4. Stress Mapping
    tree = cKDTree(dots)
    dists, indices = tree.query(dots, k=5)
    
    heatmap_data = []
    for i, p in enumerate(dots):
        # Calculate local strain/stress
        neighbor_pts = dots[indices[i, 1:]]
        avg_dist = np.mean(dists[i, 1:])
        
        strain = ((avg_dist - baseline_dist) / baseline_dist) * params['factor']
        heatmap_data.append([p[0], p[1], strain])

    # 5. Interpolation & Visualization
    pts = np.array(heatmap_data)
    grid_x, grid_y = np.mgrid[0:w:4, 0:h:4]
    grid_z = griddata(pts[:, :2], pts[:, 2], (grid_x, grid_y), method='linear', fill_value=0)
    
    # Create Heatmap Overlay
    res = cv2.resize(grid_z.T, (w, h))
    limit = max(abs(np.nanmin(res)), abs(np.nanmax(res)), 0.01)
    norm = np.clip((res + limit) / (2 * limit), 0, 1)
    color_map = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend
    hull = cv2.convexHull(dots.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    
    output = img_bgr.copy()
    mask_indices = mask > 0
    output[mask_indices] = cv2.addWeighted(color_map, 0.6, img_bgr, 0.4, 0)[mask_indices]

    # 6. Center Point Calculation
    center_idx = tree.query([w/2, h/2], k=1)[1]
    sx, sy = engine.get_plane_stress(dots[center_idx], dots[indices[center_idx, 1:]], baseline_dist)
    logs.append(f"🎯 **Center Stress X:** {sx:.2f} MPa")
    logs.append(f"🎯 **Center Stress Y:** {sy:.2f} MPa")

    return output, logs, binary_mask, dots

# ==========================================
# 3. USER INTERFACE (Streamlit)
# ==========================================
def main():
    st.set_page_config(page_title="Stress Engine v2", layout="wide")
    st.title("🔬 Advanced Stress Topography")

    with st.sidebar:
        st.header("⚙️ Configuration")
        p = {
            'blur': st.slider("Blur (Noise Reduction)", 1, 15, 7, 2),
            'thresh': st.slider("Threshold Block Size", 3, 31, 11, 2),
            'min_area': st.slider("Min Dot Pixels", 1, 50, 5),
            'manual_spacing': st.number_input("Manual Spacing (px)", 0.0),
            'adj_perc': st.slider("Zero-Point Calibration (%)", -15.0, 15.0, 0.0, 0.1),
            'modulus': st.number_input("Young's Modulus (MPa)", 51.0),
            'poisson': st.number_input("Poisson Ratio", 0.3),
            'factor': st.number_input("Strain Scale Factor", 1.0)
        }

    upl = st.file_uploader("Upload Speckle Pattern", type=['png', 'jpg'])
    
    if upl:
        img = Image.open(upl)
        if st.button("RUN CALCULATION", type="primary"):
            res_img, logs, binary, dots = run_analysis(img, p)
            
            if res_img is not None:
                c1, c2 = st.columns([2, 1])
                c1.image(res_img, channels="BGR", caption="Stress Distribution Map")
                
                with c2:
                    st.subheader("Results")
                    for l in logs: st.write(l)
                    
                    with st.expander("Diagnostics"):
                        st.image(binary, caption="Binary Mask")
                        viz_dots = np.array(img.convert('RGB'))
                        for d in dots: cv2.circle(viz_dots, tuple(d), 3, (0,255,0), -1)
                        st.image(viz_dots, caption="Detected Dots")
            else:
                st.error(logs[0])

if __name__ == "__main__":
    main()
