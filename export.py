"""Writes a minimal Tacview ACMI 2.1 (text) file from a logged trajectory."""

import numpy as np

REF_LAT, REF_LON = 0.0, 0.0
M2FT_LAT = 1.0 / 111320.0


def _lla(pos_ned):
    lat = REF_LAT + pos_ned[1] * M2FT_LAT
    lon = REF_LON + pos_ned[0] * M2FT_LAT
    alt = pos_ned[2]
    return lat, lon, alt


def write_acmi(trajectory, path):
    if not trajectory:
        return
    with open(path, "w") as f:
        f.write("FileType=text/acmi/tacview\nFileVersion=2.1\n")
        f.write(f"0,ReferenceTime=2026-01-01T00:00:00Z\n")
        for row in trajectory:
            t = row["t"]
            f.write(f"#{t:.2f}\n")
            lat, lon, alt = _lla(row["self_pos"])
            f.write(f"SELF,T={lon:.7f}|{lat:.7f}|{alt:.1f},Name=F-16,Color=Blue\n")
            lat, lon, alt = _lla(row["adv_pos"])
            f.write(f"ADV,T={lon:.7f}|{lat:.7f}|{alt:.1f},Name=F-16,Color=Red\n")