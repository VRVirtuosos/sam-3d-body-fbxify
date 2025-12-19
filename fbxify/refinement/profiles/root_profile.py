from fbxify.refinement.profiles.filter_profile import FilterProfile

ROOT_PROFILE = FilterProfile(
    max_pos_speed=1.5,
    max_pos_accel=15.0,
    max_ang_speed_deg=180.0,
    max_ang_accel_deg=1800.0,
    method="ema",
    cutoff_hz=2.0,
    root_cutoff_xy_hz=1.5,
    root_cutoff_z_hz=0.6,
)
