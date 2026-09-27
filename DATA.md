# Data documentation

Everything you need to understand and trust the dataset, without us in the room.

## At a glance

| | |
|---|---|
| **What** | Ambient ground vibration recorded with a phone lying flat on the ground, plus a GPS fix and a tilt check at each spot |
| **Where** | Five spots on the IIT Guwahati campus: Old SAC, KV Gate, Core 4 (academic complex), beside the Lecture Hall lake, and the open ground between Barak and Umiam hostels (see [sites_manifest.csv](sites_manifest.csv)) |
| **When (collection window)** | 26 September 2026, **19:57:11 to 22:49:12 IST**. Four spots between 19:57 and 20:40; Barak–Umiam between 22:43 and 22:49 |
| **Cadence** | Vibration: continuous, 99.76 readings per second, for 186–207 s per spot (16.7 min in total). GPS: about one fix per second for 15–48 s. Tilt: about two readings per second for 20–23 s. One visit per spot |
| **Row counts** | **100,212** vibration readings · **94** GPS fixes · **210** tilt readings · **15** recordings (3 per spot) |
| **Source / instrument** | One phone: vivo V2545 (Android 16), TDK icm42607 accelerometer, phyphox app 1.2.1 (RWTH Aachen University), tools "Acceleration (without g)", "Location (GPS)" and "Inclination" |
| **Format** | Original phyphox zip exports (CSV inside) in `data/raw/`; plain-CSV sample in `data/sample/`; Parquet produced by the pipeline in `data/processed/` |
| **Provenance** | Every raw file's SHA-256 checksum and wall-clock start and end time are in [data/sample/recordings.csv](data/sample/recordings.csv) |
| **Synthetic data** | **None**, anywhere in this repository |

## Per-spot collection log

| Spot | Vibration recording (IST) | Readings | Duration | GPS accuracy | Phone tilt |
|---|---|---|---|---|---|
| Old SAC | 19:57–20:00 | 20,062 | 201 s | ±2.6 m (satellite fix) | 1.1° |
| KV Gate | 20:12–20:15 | 20,555 | 206 s | ±1.8 m (satellite fix) | 5.6° (above our 5° limit; flagged) |
| Core 4 | 20:21–20:25 | 20,366 | 204 s | ±2.4 m (satellite fix) | 1.5° |
| Lecture Hall lake | 20:34–20:37 | 20,644 | 207 s | ±1.7 m (satellite fix) | 3.6° |
| Barak–Umiam | 22:43–22:46 | 18,585 | 186 s | ±100 m (network fix only) | 0.8° |

## Observed vs inferred vs synthetic

| Type | What it covers | Where |
|---|---|---|
| **Observed** (measured by the phone) | Three-direction acceleration, GPS fixes, tilt angles, recording start and stop times, device details | `data/raw/`, `data/sample/`, `data/processed/` |
| **Provided by us** (human input) | A plain-English description of each spot and a rough ground-type label. The labels are our best guess, not surveyed | `sites_manifest.csv` |
| **Inferred** (calculated from the observations) | Which 20-second pieces were clean, the sideways-to-up-down curves, the candidate rhythm and strength, the nine SESAME rulebook checks, the hiss check and verdict, the vibration upper bounds, the AI classifier result, and the longer-recording projection (figure 13) | `outputs/summary.csv`, `outputs/insights.json`, `outputs/figures/` |
| **Synthetic** | **None.** The only random numbers are inside the code's self-test (`--selfcheck`), which is never saved or plotted | n/a |

## How AI was used

1. **A test of our own data, not a product.** A logistic-regression classifier (scikit-learn) learned the vibration "fingerprint" of 84 quiet 10-second windows (the strength of shaking in 14 frequency bands, in three directions). It was tested on blocks of time it had not seen, so it could not cheat by remembering neighbouring moments. If the ground really differed between spots, it should have told them apart using the rhythms that matter for buildings (0.8–8 vibrations per second). It scored **21%, against 20% for random guessing**. Using all frequencies it scored 26%, and the only spot it could recognise was Barak–Umiam, because of a pump running nearby.
2. **Automated quality control.** A burst detector (STA/LTA), a loudness check, the nine SESAME checks and a hiss check together decide each spot's verdict, without anyone judging the curves by eye.
3. **Coding help.** AI coding assistants helped write and review parts of the code. All method choices and results were checked by the team.

## Processing summary

1. Recognise each file by keyword.
2. Remove duplicate timestamps (none were found).
3. Drop the first and last 5 s of each recording.
4. Reject disturbed stretches (burst trigger with 1 s / 10 s windows and threshold 2.5, plus pieces louder than 3× the spot's median).
5. Cut into 20-second pieces with 50% overlap.
6. Take the frequency spectrum of each piece, smooth it (Konno–Ohmachi, b = 40), and divide sideways by up-and-down shaking.
7. Average the pieces in log space.
8. Run the nine SESAME (2004) checks.
9. Check that the signal is above the phone's own hiss (about 1 mm/s² per √Hz, measured from these recordings).
10. Issue a verdict per spot.

**Result:** 80 of 93 pieces kept; all five spots `unresolved_sensor_floor`.

## Known gaps and biases

- **The phone's hiss is the main limit for this first round.** In 3½-minute night recordings it is 38–696 times stronger than natural ground trembling between 0.5 and 10 vibrations per second, so no ground rhythm could be measured yet. Longer and multi-phone recordings make the hiss steadier and lower, letting fainter signals show: a projection from our measured noise gives about 0.39× the hiss for 30 minutes with one phone, and 0.14× for 2 hours with four phones side by side (figure 13). That could reach busier or softer spots at the faster rhythms. The quietest ground at slow rhythms still needs a proper sensor.
- Recordings are 3–3½ minutes, shorter than the 10–15 minutes the guidelines suggest.
- One phone, one evening, five spots, one visit each: no daytime, weekday or seasonal comparison.
- The hiss level was estimated from these same recordings, not from a separate test with the phone on a still surface indoors.
- The ground-type labels are our own rough guesses, and the Barak–Umiam position is only accurate to about ±100 m.
- The vibration upper bounds (quieter than an operating-theatre limit) come from quiet night-time moments only and are for screening, not certification.
