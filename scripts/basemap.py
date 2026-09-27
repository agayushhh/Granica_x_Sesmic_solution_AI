"""OpenStreetMap tile stitching (stdlib only), shared by make_figures.py and make_deck.py."""
import io
import math
import urllib.request
import numpy as np
import matplotlib.pyplot as plt


def fetch_osm(lat_rng, lon_rng, z=16):
    """Returns (rgb image, (lon_left, lon_right, lat_bottom, lat_top)), or None if offline.
    Attribution '© OpenStreetMap contributors' must appear wherever the image is shown."""
    n = 2 ** z

    def tile(lat, lon):
        return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n

    def untile(x, y):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))), x / n * 360 - 180

    x0, y1 = tile(lat_rng[0], lon_rng[0])
    x1, y0 = tile(lat_rng[1], lon_rng[1])
    xs, ys = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    try:
        rows = []
        for ty in ys:
            row = []
            for tx in xs:
                req = urllib.request.Request(f"https://tile.openstreetmap.org/{z}/{tx}/{ty}.png",
                                             headers={"User-Agent": "iitg-hvsr-survey/1.0 (hackathon figure)"})
                row.append(plt.imread(io.BytesIO(urllib.request.urlopen(req, timeout=10).read()), format="png")[..., :3])
            rows.append(np.hstack(row))
    except Exception as e:
        print("  (no basemap:", e, ")")
        return None
    la_top, lo_left = untile(xs[0], ys[0])
    la_bot, lo_right = untile(xs[-1] + 1, ys[-1] + 1)
    return np.vstack(rows), (lo_left, lo_right, la_bot, la_top)
