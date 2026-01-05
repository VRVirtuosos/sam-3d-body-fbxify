from fbxify.refinement.profiles.filter_profile import FilterProfile

ARMS_PROFILE = FilterProfile(
    max_pos_speed=1.0,
    max_pos_accel=10.0,
    max_ang_speed_deg=360.0,
    max_ang_accel_deg=3600.0,
    method="one_euro",
    one_euro_min_cutoff=2.0,  # Higher than default (1.0) to reduce lag
    one_euro_beta=0.6,        # Higher than default (0.3) to follow motion better
    one_euro_d_cutoff=1.0,    # Higher than default (0.8)
)

