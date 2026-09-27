# Schema

Every file in this repository, what each column means, and its unit. `location_id` links everything together.

| `location_id` | Spot |
|---|---|
| `old_sac` | Open ground near the Old SAC (Student Activity Centre) |
| `kv_gate` | Ground near the Kendriya Vidyalaya (KV) gate |
| `core_4` | Academic complex, near Core 4 |
| `lecture_hall_lake` | Beside the lake next to the Lecture Hall complex |
| `barak_umium` | Open ground between Barak and Umiam hostels |

---

## 1. Raw phone exports: `data/raw/*.zip`

There are three zips per spot, exactly as phyphox saved them. The file names were typed by hand in the field, so they vary; the code matches them by keyword. Each zip also holds `meta/time.csv` (wall-clock start and stop) and `meta/device.csv` (phone and sensor details).

**Vibration: `… Acceleration (without g).zip` → `Raw Data.csv`**

| Column | Unit | Meaning |
|---|---|---|
| `Time (s)` | s | Time since the recording started |
| `Linear Acceleration x (m/s^2)` | m/s² | Shaking along the phone's x axis (gravity removed) |
| `Linear Acceleration y (m/s^2)` | m/s² | Shaking along the phone's y axis |
| `Linear Acceleration z (m/s^2)` | m/s² | Shaking along the phone's z axis, which is vertical when the phone lies flat |
| `Absolute acceleration (m/s^2)` | m/s² | Overall size of the shaking |

**Location: `… Location (GPS).zip` → `Raw Data.csv`**

`Time (s)`, `Latitude (°)`, `Longitude (°)`, `Altitude (m)`, `Altitude WGS84 (m)`, `Speed (m/s)`, `Direction (°)`, `Distance (km)`, `Horizontal Accuracy (m)`, `Vertical Accuracy (m)`, `Satellites`. A `Satellites` value of −1 means a rough network-based fix, not a satellite fix.

**Tilt: `… Inclination.zip` → `Flat.csv`** (the other CSVs in this zip show the same reading in other orientations)

`t (s)`, `Tilt up/down (deg)`, `Tilt left/right (deg)`

---

## 2. Plain-CSV sample: `data/sample/`

**`accel_sample.csv`**: the first 200 vibration readings from each spot (1,000 rows)

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `t_s` | float | s | Time since that spot's recording started |
| `ax`, `ay`, `az` | float | m/s² | Shaking along x, y and z (z = up-and-down) |
| `a_abs` | float | m/s² | Overall size of the shaking |
| `sta_lta_quiet` | bool | – | *Inferred:* `True` if this moment passed the disturbance checks. `False` for the first and last 5 s and during bursts |
| `location_id` | string | – | Which spot |

**`gps.csv`**: all 94 GPS fixes. `t_s, lat, lon (°), alt, alt_wgs84 (m), speed (m/s), direction (°), distance (km), h_acc, v_acc (m), satellites, location_id`

**`inclination.csv`**: all 210 tilt readings. `t_s, tilt_updown, tilt_leftright (°), location_id`

**`recordings.csv`**: provenance, one row per raw file (15 rows)

| Column | Meaning |
|---|---|
| `location_id`, `kind` | Spot, and `accel` / `gps` / `incl` |
| `source_file`, `sha256` | Original file name and its checksum |
| `start_ist`, `end_ist` | Wall-clock start and end (India Standard Time) |
| `device`, `android_release`, `phyphox_version`, `accelerometer` | Phone, operating system, app version and sensor chip |

---

## 3. Processed data (created by the pipeline): `data/processed/`

`accel.parquet` (100,212 rows), `gps.parquet` (94), `inclination.parquet` (210) and `recordings.csv` have the same columns as the sample above, for all the data. `sample/accel_sample.csv` is regenerated each run.

---

## 4. Results: `outputs/summary.csv` (one row per spot)

| Column | Meaning |
|---|---|
| `start_ist`, `sample_rate_hz`, `duration_s`, `n_samples` | When, how fast, how long, and how many readings |
| `f0_hz`, `a0` | *Candidate* ground rhythm (vibrations per second) and its strength: the highest point of the curve. **Not a finding when the verdict is `unresolved_sensor_floor`** |
| `log_std_at_f0`, `sigma_f_hz` | How much that point varies from piece to piece |
| `n_windows_used`, `n_windows_total` | Clean 20-second pieces kept, out of all pieces |
| `R1_…`, `R2_…`, `R3_…` | The three SESAME reliability checks (True/False) |
| `C1_…` to `C6_…` | The six SESAME "clear peak" checks (True/False) |
| `sesame_reliability_passed`, `sesame_clarity_passed` | How many of the 3 and the 6 passed |
| `band_frac_above_floor` | Share of the 0.5–15 Hz range where all three directions are clearly above the phone's hiss |
| `snr_db_at_f0` | Signal above the phone's hiss at the candidate rhythm, in decibels |
| `confidence` | **The verdict:** `high` / `medium` / `low`, or `unresolved_sensor_floor` when the phone's hiss is louder than the ground (all five spots) |
| `lat`, `lon`, `gps_horizontal_accuracy_m`, `gps_satellites` | Best GPS fix |
| `mean_tilt_deg` | Average tilt of the phone |
| `has_gps`, `has_inclination` | Whether those recordings exist |
| `description`, `zone_label`, `notes` | Copied from `sites_manifest.csv` |

**`outputs/insights.json`**: the key numbers quoted in the slides (gap between the hiss and natural ground, classifier scores, vibration upper bounds, counts).

**`outputs/figures/`**: 12 charts, numbered in the order they appear in the story.
