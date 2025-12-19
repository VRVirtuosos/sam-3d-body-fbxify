from fbxify.refinement.refinement_config import RefinementConfig
import re
import math

# ============================================================================
# Vector and Math Utilities
# ============================================================================

def norm(v):
    """Compute L2 norm of a 3D vector."""
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def dot(v1, v2):
    """Dot product of two vectors (can be 3D or 4D for quaternions)."""
    return sum(a * b for a, b in zip(v1, v2))

def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi

def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * math.pi / 180.0

def dot4(q1, q2):
    """Dot product of two quaternions."""
    return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]

def neg4(q):
    """Negate a quaternion."""
    return [-q[0], -q[1], -q[2], -q[3]]

def quat_normalize(q):
    """Normalize a quaternion."""
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n > 1e-10:
        return [q[0]/n, q[1]/n, q[2]/n, q[3]/n]
    return [1.0, 0.0, 0.0, 0.0]

# ============================================================================
# Quaternion Utilities
# ============================================================================

def quat_from_R(R):
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    R: [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    Returns: [w, x, y, z]
    """
    # Trace-based method for numerical stability
    trace = R[0][0] + R[1][1] + R[2][2]
    
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2][1] - R[1][2]) / s
        y = (R[0][2] - R[2][0]) / s
        z = (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        w = (R[2][1] - R[1][2]) / s
        x = 0.25 * s
        y = (R[0][1] + R[1][0]) / s
        z = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        w = (R[0][2] - R[2][0]) / s
        x = (R[0][1] + R[1][0]) / s
        y = 0.25 * s
        z = (R[1][2] + R[2][1]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
        w = (R[1][0] - R[0][1]) / s
        x = (R[0][2] + R[2][0]) / s
        y = (R[1][2] + R[2][1]) / s
        z = 0.25 * s
    
    # Normalize
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n > 1e-10:
        return [w/n, x/n, y/n, z/n]
    return [1.0, 0.0, 0.0, 0.0]

def R_from_quat(q):
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    q: [w, x, y, z]
    Returns: [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    """
    w, x, y, z = q
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    return [
        [w2 + x2 - y2 - z2, 2*(xy - wz), 2*(wy + xz)],
        [2*(wz + xy), w2 - x2 + y2 - z2, 2*(yz - wx)],
        [2*(xz - wy), 2*(wx + yz), w2 - x2 - y2 + z2]
    ]

def quat_mul(q1, q2):
    """Multiply two quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]

def quat_inv(q):
    """Inverse of a quaternion (conjugate for unit quaternions)."""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n > 1e-10:
        return [w/n, -x/n, -y/n, -z/n]
    return [1.0, 0.0, 0.0, 0.0]

def quat_angle(q):
    """
    Get the rotation angle (in radians) from a quaternion.
    For a unit quaternion q = [cos(θ/2), sin(θ/2)*axis], returns θ.
    """
    w = q[0]
    # Clamp to [-1, 1] for numerical stability
    w = max(-1.0, min(1.0, w))
    return 2.0 * math.acos(abs(w))

def slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.
    t: interpolation parameter [0, 1]
    """
    # Ensure shortest path
    dot_q = dot(q1, q2)
    if dot_q < 0:
        q2 = [-q2[0], -q2[1], -q2[2], -q2[3]]
        dot_q = -dot_q
    
    # Clamp for numerical stability
    dot_q = max(-1.0, min(1.0, dot_q))
    
    theta = math.acos(dot_q)
    if abs(theta) < 1e-6:
        # Quaternions are very close, use linear interpolation
        return [q1[i] + t * (q2[i] - q1[i]) for i in range(4)]
    
    sin_theta = math.sin(theta)
    w1 = math.sin((1 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    
    return [w1 * q1[i] + w2 * q2[i] for i in range(4)]

def quat_log(q):
    """
    Logarithm map: quaternion -> tangent space (axis-angle representation).
    Returns: [x, y, z] (angular velocity vector)
    """
    w = q[0]
    x, y, z = q[1], q[2], q[3]
    
    # Clamp w for numerical stability
    w = max(-1.0, min(1.0, w))
    
    angle = math.acos(abs(w))
    if angle < 1e-6:
        return [0.0, 0.0, 0.0]
    
    sin_angle = math.sin(angle)
    if sin_angle < 1e-6:
        return [0.0, 0.0, 0.0]
    
    scale = 2.0 * angle / sin_angle
    if w < 0:
        scale = -scale
    
    return [scale * x, scale * y, scale * z]

def quat_exp(v):
    """
    Exponential map: tangent space -> quaternion (axis-angle -> quaternion).
    v: [x, y, z] (angular velocity vector)
    Returns: [w, x, y, z]
    """
    angle = norm(v)
    if angle < 1e-6:
        return [1.0, 0.0, 0.0, 0.0]
    
    half_angle = 0.5 * angle
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)
    
    scale = sin_half / angle
    return [cos_half, scale * v[0], scale * v[1], scale * v[2]]

# ============================================================================
# Filter Functions
# ============================================================================

def butterworth_lowpass(v_series, cutoff_hz, dt, order=2):
    """
    Apply Butterworth lowpass filter to a vector series using biquad (2nd order IIR).
    v_series: [T][3] list of vectors
    cutoff_hz: cutoff frequency in Hz
    dt: time step in seconds
    order: filter order (default 2, only 2 is currently implemented)
    Returns: filtered [T][3] list
    
    This implements a true 2nd-order Butterworth filter with:
    - Maximally flat passband response
    - 40dB/decade rolloff (vs 20dB for first-order/EMA)
    - Proper biquad coefficients via bilinear transform
    """
    T = len(v_series)
    if T < 3:
        return v_series
    
    # Convert cutoff frequency to radians per second
    wc = 2.0 * math.pi * cutoff_hz
    
    # Pre-warp frequency for bilinear transform to compensate for frequency warping
    # This ensures the digital filter matches the analog cutoff frequency
    tan_wc_dt_2 = math.tan(wc * dt / 2.0)
    
    # For 2nd order Butterworth: H(s) = wc^2 / (s^2 + sqrt(2)*wc*s + wc^2)
    # Using bilinear transform: s = (2/dt) * (z-1)/(z+1)
    # After algebra, we get biquad coefficients:
    
    # Normalize by tan term
    k = tan_wc_dt_2
    k2 = k * k
    
    # Biquad coefficients for 2nd order Butterworth lowpass
    # Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
    a0 = 1.0 + math.sqrt(2) * k + k2
    b0 = k2 / a0
    b1 = 2.0 * k2 / a0
    b2 = k2 / a0
    a1 = 2.0 * (k2 - 1.0) / a0
    a2 = (1.0 - math.sqrt(2) * k + k2) / a0
    
    # Initialize output and filter state (previous values for IIR)
    out = [list(v_series[0])]
    
    # Filter state: [x[n-1], x[n-2], y[n-1], y[n-2]] for each component
    # x = input, y = output
    state = [[[0.0, 0.0, 0.0, 0.0] for _ in range(3)]]
    
    for t in range(1, T):
        filtered = [0.0, 0.0, 0.0]
        for i in range(3):
            # Current input
            x = v_series[t][i]
            
            # Get previous state
            if t == 1:
                x_prev = v_series[0][i]
                x_prev2 = v_series[0][i]
                y_prev = out[0][i]
                y_prev2 = out[0][i]
            else:
                x_prev = v_series[t-1][i]
                x_prev2 = v_series[t-2][i]
                y_prev = out[t-1][i]
                y_prev2 = out[t-2][i]
            
            # Biquad filter: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
            y = (b0 * x + b1 * x_prev + b2 * x_prev2 - 
                 a1 * y_prev - a2 * y_prev2)
            
            filtered[i] = y
        
        out.append(filtered)
    
    return out

def butterworth_quat(q_series, cutoff_hz, dt, order=2):
    """
    Apply Butterworth filter to quaternion series via tangent space.
    q_series: [T] list of quaternions [w, x, y, z]
    cutoff_hz: cutoff frequency in Hz
    dt: time step in seconds
    Returns: filtered [T] list of quaternions
    """
    T = len(q_series)
    if T < 3:
        return q_series
    
    # Convert to tangent space (log map)
    tangent_vectors = []
    for t in range(T):
        if t == 0:
            tangent_vectors.append([0.0, 0.0, 0.0])
        else:
            q_rel = quat_mul(quat_inv(q_series[t-1]), q_series[t])
            tangent_vectors.append(quat_log(q_rel))
    
    # Filter in tangent space
    filtered_tangent = butterworth_lowpass(tangent_vectors, cutoff_hz, dt, order)
    
    # Re-integrate back to quaternions
    out = [q_series[0]]
    for t in range(1, T):
        q_delta = quat_exp(filtered_tangent[t])
        q_new = quat_mul(out[t-1], q_delta)
        out.append(q_new)
    
    return out

class OneEuroFilter:
    """One Euro filter for scalar values."""
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, dt=1.0/30.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.dt = dt
        self.x_prev = None
        self.dx_prev = 0.0
    
    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        
        # Estimate derivative
        dx = (x - self.x_prev) / self.dt
        
        # Smooth derivative
        edx = self._smooth(dx, self.dx_prev, self.d_cutoff)
        self.dx_prev = edx
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        
        # Smooth signal
        x_filtered = self._smooth(x, self.x_prev, cutoff)
        self.x_prev = x_filtered
        
        return x_filtered
    
    def _smooth(self, x, x_prev, cutoff):
        """Simple exponential smoothing."""
        te = 1.0 / (2.0 * math.pi * cutoff)
        alpha = self.dt / (te + self.dt)
        return alpha * x + (1.0 - alpha) * x_prev

def ema_filter_vec3(v_series, cutoff_hz, dt):
    """
    Apply Exponential Moving Average (EMA) lowpass filter to a vector series.
    v_series: [T][3] list of vectors
    cutoff_hz: cutoff frequency in Hz
    dt: time step in seconds
    Returns: filtered [T][3] list
    
    EMA is a first-order IIR filter with 20dB/decade rolloff.
    """
    T = len(v_series)
    if T == 0:
        return v_series
    
    # Calculate alpha from cutoff frequency
    # For a first-order lowpass: alpha = 1 - exp(-2*pi*fc*dt)
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
    
    out = [list(v_series[0])]
    for t in range(1, T):
        filtered = [out[t-1][i] + alpha * (v_series[t][i] - out[t-1][i]) for i in range(3)]
        out.append(filtered)
    
    return out

def ema_filter_quat(q_series, cutoff_hz, dt):
    """
    Apply EMA filter to quaternion series via SLERP.
    q_series: [T] list of quaternions [w, x, y, z]
    cutoff_hz: cutoff frequency in Hz
    dt: time step in seconds
    Returns: filtered [T] list of quaternions
    
    EMA in SO(3) is performed via spherical linear interpolation (SLERP).
    """
    T = len(q_series)
    if T == 0:
        return q_series
    
    # Calculate alpha from cutoff frequency
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
    
    out = [q_series[0]]
    for t in range(1, T):
        # EMA in SO(3) via SLERP toward new sample
        out.append(slerp(out[t-1], q_series[t], alpha))
    
    return out

def one_euro_filter_vec3(v_series, dt, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    """
    Apply One Euro filter to a 3D vector series.
    v_series: [T][3] list of vectors
    dt: time step in seconds
    min_cutoff: minimum cutoff frequency
    beta: speed coefficient
    d_cutoff: derivative cutoff frequency
    Returns: filtered [T][3] list
    """
    T = len(v_series)
    if T == 0:
        return v_series
    
    # Create separate filters for each component
    filters = [OneEuroFilter(min_cutoff, beta, d_cutoff, dt) for _ in range(3)]
    
    out = []
    for t in range(T):
        filtered = [filters[i](v_series[t][i]) for i in range(3)]
        out.append(filtered)
    
    return out

def fix_quat_hemisphere(qs):
    out = [qs[0]]
    for t in range(1, len(qs)):
        out.append(qs[t] if dot4(out[t-1], qs[t]) >= 0 else neg4(qs[t]))
    return out

def one_euro_filter_quat(q_series, dt, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    T = len(q_series)
    if T < 2:
        return q_series

    q_series = fix_quat_hemisphere(q_series)

    tangent = [[0.0,0.0,0.0]]
    for t in range(1, T):
        q_rel = quat_mul(quat_inv(q_series[t-1]), q_series[t])
        tangent.append(quat_log(q_rel))

    filtered = one_euro_filter_vec3(tangent, dt, min_cutoff, beta, d_cutoff)

    out = [quat_normalize(q_series[0])]
    for t in range(1, T):
        q_delta = quat_exp(filtered[t])
        q_new = quat_mul(out[t-1], q_delta)
        out.append(quat_normalize(q_new))

    return out



# ============================================================================
# Refinement Manager
# ============================================================================

class RefinementManager:
    """
    RefinementManager is a class that applies refinement and mocap-style smoothing to the animation.
    """
    def __init__(self, config: RefinementConfig = None):
        if config is None:
            # use default config
            config = RefinementConfig()
        self.configure(config)

    def configure(self, config: RefinementConfig):
        self.config = config
        self.dt = 1.0 / config.fps  # time step in seconds
    
    def _calculate_vector_change_percent(self, v_original, v_refined):
        """
        Calculate the percentage change between original and refined vector series.
        Returns the average percentage change in magnitude.
        """
        if len(v_original) == 0 or len(v_refined) == 0:
            return 0.0
        
        total_change = 0.0
        total_original_mag = 0.0
        count = 0
        
        for t in range(min(len(v_original), len(v_refined))):
            orig_mag = norm(v_original[t])
            refined_mag = norm(v_refined[t])
            
            if orig_mag > 1e-6:  # Avoid division by zero
                change = abs(refined_mag - orig_mag) / orig_mag * 100.0
                total_change += change
                total_original_mag += orig_mag
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_change / count
    
    def _identity_matrix(self):
        """Return a 3x3 identity matrix."""
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    
    def _convert_to_list(self, value):
        """Convert numpy array or other types to list."""
        if hasattr(value, 'tolist'):
            return value.tolist()
        if hasattr(value, 'shape'):
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.tolist()
        return value
    
    def _parse_3x3_nested_matrix(self, rot_list):
        """Parse a 3x3 nested list matrix: [[a,b,c], [d,e,f], [g,h,i]]."""
        if not isinstance(rot_list, list) or len(rot_list) != 3:
            return None
        if not isinstance(rot_list[0], (list, tuple)) or len(rot_list[0]) != 3:
            return None
        return [[float(rot_list[i][j]) for j in range(3)] for i in range(3)]
    
    def _parse_3x3_flattened_matrix(self, rot_list):
        """Parse a flattened 9-element list into 3x3 matrix: [a,b,c,d,e,f,g,h,i]."""
        if not isinstance(rot_list, list) or len(rot_list) != 9:
            return None
        return [[float(rot_list[i*3 + j]) for j in range(3)] for i in range(3)]
    
    def _parse_rotation_matrix(self, rot_t):
        """Parse a rotation matrix from various formats, return 3x3 list or None."""
        rot_t = self._convert_to_list(rot_t)
        
        if not isinstance(rot_t, list):
            return None
        
        # Try nested 3x3 format first
        nested = self._parse_3x3_nested_matrix(rot_t)
        if nested is not None:
            return nested
        
        # Try flattened 9-element format
        flattened = self._parse_3x3_flattened_matrix(rot_t)
        if flattened is not None:
            return flattened
        
        return None
    
    def _deep_copy_rotation_series(self, rot):
        """Create a deep copy of rotation series, handling various formats."""
        rot_original = []
        for t in range(len(rot)):
            try:
                parsed = self._parse_rotation_matrix(rot[t])
                rot_original.append(parsed if parsed is not None else self._identity_matrix())
            except (TypeError, IndexError) as e:
                print(f"Warning: Could not parse rotation matrix at frame {t}: {e}, using identity")
                rot_original.append(self._identity_matrix())
        return rot_original
    
    def _calculate_rotation_change_percent(self, R_original, R_refined):
        """
        Calculate the percentage change between original and refined rotation series.
        Returns the average angular change in degrees.
        """
        if len(R_original) == 0 or len(R_refined) == 0:
            return 0.0
        
        q_original = [quat_from_R(R) for R in R_original]
        q_refined = [quat_from_R(R) for R in R_refined]
        q_original = fix_quat_hemisphere(q_original)
        q_refined = fix_quat_hemisphere(q_refined)
        
        total_angle_change = 0.0
        count = 0
        
        for t in range(min(len(q_original), len(q_refined))):
            # Calculate relative rotation between original and refined
            q_rel = quat_mul(quat_inv(q_original[t]), q_refined[t])
            angle_deg = rad2deg(quat_angle(q_rel))
            total_angle_change += angle_deg
            count += 1
        
        if count == 0:
            return 0.0
        
        return total_angle_change / count

    def apply(self, root_node, root_motion=None):
        """
        Apply refinement and mocap-style smoothing to the animation.
        - root_node: the root node of the animation (can be None if only refining root_motion)
        - root_motion: the root motion of the animation
        - returns: the refined root node (full joint_to_bone_mapping) and root motion
        """
        if root_node is not None:
            self._refine_node_recursive(root_node)

        if root_motion is not None and self.config.do_root_motion_fix:
            root_motion = self._root_stabilization(root_motion)

        return root_node, root_motion

    def _refine_node_recursive(self, node):
        """
        Refine a node recursively.
        - node: the node to refine
        - returns: none - mutable data is modified in place
        """
        prof = self._profile_for(node["name"])

        mapping = node.get("mapping") or node  # depending on your JSON shape
        method = mapping.get("method")
        data = mapping.get("data", {})

        if method == "direct_rotation":
            if "rotation" in data:
                mats = data["rotation"]  # [T][3][3]
                mats = self._process_rotation_series(mats, prof, bone_name=node["name"])
                data["rotation"] = mats
        elif method == "keypoint_with_global_rot_roll":
            if "dir_vector" in data:
                v = data["dir_vector"]   # [T][3]
                v = self._process_vector_series(v, prof, bone_name=f"{node['name']}.dir_vector")
                data["dir_vector"] = v

            if "roll_vector" in data:
                mats = data["roll_vector"]  # [T][3][3]
                mats = self._process_rotation_series(mats, prof, bone_name=f"{node['name']}.roll_vector")
                data["roll_vector"] = mats
        else:
            # Future mapping methods?
            print(f"  WARNING: While refinining [{node['name']}] action, the bone has unknown mapping method: {method}")

        # Recursively process children
        if "children" in node:
            for child in node["children"]:
                self._refine_node_recursive(child)

        return node

    def _process_vector_series(self, v_series, prof, bone_name=None):
        v_original = [list(v) for v in v_series]  # Deep copy for comparison
        v = v_series[:]  # [T][3]

        if self.config.do_spike_fix:
            v = self._despike_vector(v, prof)

        if self.config.do_vector_smoothing:
            v = self._smooth_vector(v, prof)
        
        # Calculate and report percentage change
        if bone_name:
            change_percent = self._calculate_vector_change_percent(v_original, v)
            if change_percent > 0.01:  # Only report if there's meaningful change
                print(f"  {bone_name}: adjusted vector by {change_percent:.2f}%")

        return v

    def _despike_vector(self, v, prof):
        # velocity/accel based outlier removal
        T = len(v)
        if T < 3:
            return v

        # precompute vel/acc
        vel = [[0.0, 0.0, 0.0] for _ in range(T)]
        acc = [[0.0, 0.0, 0.0] for _ in range(T)]
        for t in range(1, T):
            vel[t] = [(v[t][i] - v[t-1][i]) / self.dt for i in range(3)]
        for t in range(2, T):
            acc[t] = [(vel[t][i] - vel[t-1][i]) / self.dt for i in range(3)]

        for t in range(1, T-1):
            speed = norm(vel[t])
            a = norm(acc[t])

            is_spike = (speed > prof.max_pos_speed and a > prof.max_pos_accel)

            # classic "single frame pop": neighbors are consistent but middle isn't
            if is_spike:
                v[t] = [0.5 * (v[t-1][i] + v[t+1][i]) for i in range(3)]

        return v

    def _smooth_vector(self, v, prof):
        if prof.method == "ema":
            return ema_filter_vec3(v, cutoff_hz=prof.cutoff_hz, dt=self.dt)

        if prof.method == "butterworth":
            # placeholder: you’d design biquad coefficients for cutoff_hz
            return butterworth_lowpass(v, cutoff_hz=prof.cutoff_hz, dt=self.dt)

        if prof.method == "one_euro":
            return one_euro_filter_vec3(
                v, dt=self.dt,
                min_cutoff=prof.one_euro_min_cutoff,
                beta=prof.one_euro_beta,
                d_cutoff=prof.one_euro_d_cutoff
            )

        return v

    def _process_rotation_series(self, R_series, prof, bone_name=None):
        # R_series: [T][3][3]
        R_original = [[[R[i][j] for j in range(3)] for i in range(3)] for R in R_series]  # Deep copy
        q = [quat_from_R(R) for R in R_series]   # [T] quats
        q = fix_quat_hemisphere(q)

        if self.config.do_spike_fix:
            q = self._despike_rotation(q, prof)

        if self.config.do_rotation_smoothing:
            q = self._smooth_rotation(q, prof)

        # back to matrices
        R_refined = [R_from_quat(qt) for qt in q]
        
        # Calculate and report percentage change
        if bone_name:
            change_deg = self._calculate_rotation_change_percent(R_original, R_refined)
            if change_deg > 0.1:  # Only report if there's meaningful change (>0.1 degrees)
                print(f"  {bone_name}: adjusted rotation by {change_deg:.2f}° (avg)")
        
        return R_refined

    def _despike_rotation(self, q, prof):
        T = len(q)
        if T < 3:
            return q

        ang_vel = [0.0] * T
        ang_acc = [0.0] * T

        for t in range(1, T):
            dq = quat_mul(quat_inv(q[t-1]), q[t])
            angle_deg = rad2deg(quat_angle(dq))          # shortest angle
            ang_vel[t] = angle_deg / self.dt

        for t in range(2, T):
            ang_acc[t] = (ang_vel[t] - ang_vel[t-1]) / self.dt

        for t in range(1, T-1):
            is_spike = (ang_vel[t] > prof.max_ang_speed_deg and
                        ang_acc[t] > prof.max_ang_accel_deg)

            if is_spike:
                # replace with slerp neighbor midpoint
                q[t] = slerp(q[t-1], q[t+1], 0.5)

        return q

    def _smooth_rotation(self, q, prof):
        if prof.method == "ema":
            return ema_filter_quat(q, cutoff_hz=prof.cutoff_hz, dt=self.dt)

        if prof.method == "one_euro":
            # OneEuro on rotation *vector* in tangent space:
            # r_t = log( inv(q_prev) * q_t ) / dt  (angular velocity in local frame)
            return one_euro_filter_quat(
                q, dt=self.dt,
                min_cutoff=prof.one_euro_min_cutoff,
                beta=prof.one_euro_beta,
                d_cutoff=prof.one_euro_d_cutoff
            )

        if prof.method == "butterworth":
            # Butterworth on tangent vectors (log map), then re-integrate.
            return butterworth_quat(q, cutoff_hz=prof.cutoff_hz, dt=self.dt)

        return q

    def _profile_for(self, bone_name):
        # very rough wildcard matching
        for pattern, prof in self.config.profiles.items():
            if self._match(pattern, bone_name):
                return prof
        return self.config.profiles["*"]

    def _match(self, pattern, bone_name):
        bone_name = bone_name.lower()
        regex = re.compile(pattern.replace("*", ".*").lower())
        return regex.match(bone_name) is not None

    def _root_stabilization(self, root_motion):
        """
        Stabilize root motion to reduce jitter and unwanted movement.
        root_motion: dict with keys like "translation" [T][3] and "rotation" [T][3][3]
        Returns: stabilized root_motion dict
        """
        if root_motion is None:
            return root_motion
        
        prof = self.config.profiles.get("root", self.config.profiles["*"])
        stabilized = {}
        
        # Stabilize translation (position)
        if "translation" in root_motion:
            trans = root_motion["translation"]  # [T][3]
            trans_original = [[t[i] for i in range(3)] for t in trans]  # Deep copy for comparison
            T = len(trans)
            
            if T > 0:
                # Apply different cutoffs for XY (horizontal) vs Z (vertical)
                # For root motion, we want different smoothing for horizontal vs vertical
                if prof.method == "one_euro":
                    # Filter X and Y with horizontal cutoff
                    filter_x = OneEuroFilter(
                        min_cutoff=prof.root_cutoff_xy_hz,
                        beta=prof.one_euro_beta,
                        d_cutoff=prof.one_euro_d_cutoff,
                        dt=self.dt
                    )
                    filter_y = OneEuroFilter(
                        min_cutoff=prof.root_cutoff_xy_hz,
                        beta=prof.one_euro_beta,
                        d_cutoff=prof.one_euro_d_cutoff,
                        dt=self.dt
                    )
                    # Filter Z with vertical cutoff (typically lower for less jitter)
                    filter_z = OneEuroFilter(
                        min_cutoff=prof.root_cutoff_z_hz,
                        beta=prof.one_euro_beta,
                        d_cutoff=prof.one_euro_d_cutoff,
                        dt=self.dt
                    )
                    
                    filtered_trans = []
                    for t in range(T):
                        filtered_trans.append([
                            filter_x(trans[t][0]),
                            filter_y(trans[t][1]),
                            filter_z(trans[t][2])
                        ])
                    stabilized["translation"] = filtered_trans
                else:
                    # For other methods, use profile cutoffs (standard processing)
                    filtered = self._process_vector_series(trans, prof)
                    stabilized["translation"] = filtered
                
                # Calculate and report percentage change for root translation
                change_percent = self._calculate_vector_change_percent(trans_original, stabilized["translation"])
                if change_percent > 0.01:  # Only report if there's meaningful change
                    print(f"root_stabilization: adjusted root translation by {change_percent:.2f}%")
        
        # Stabilize rotation
        if "rotation" in root_motion:
            rot = root_motion["rotation"]  # [T][3][3]
            rot_original = self._deep_copy_rotation_series(rot)
            
            # Process without bone_name to avoid duplicate message
            stabilized["rotation"] = self._process_rotation_series(rot, prof, bone_name=None)
            
            # Calculate and report percentage change for root rotation
            change_deg = self._calculate_rotation_change_percent(rot_original, stabilized["rotation"])
            if change_deg > 0.1:  # Only report if there's meaningful change
                print(f"root_stabilization: adjusted root rotation by {change_deg:.2f}° (avg)")
        
        # Copy any other fields
        for key in root_motion:
            if key not in stabilized:
                stabilized[key] = root_motion[key]
        
        return stabilized



