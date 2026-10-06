"""Writes a minimal Tacview ACMI 2.1 (text) file from a logged trajectory."""

import numpy as np

REF_LAT, REF_LON = 0.0, 0.0
M2FT_LAT = 1.0 / 111320.0


def _lla(pos_ned):
    lat = REF_LAT + pos_ned[1] * M2FT_LAT
    lon = REF_LON + pos_ned[0] * M2FT_LAT
    alt = pos_ned[2]
    return lat, lon, alt


def _transform(pos, att):
    """ACMI T= field: lon|lat|alt, plus |roll|pitch|yaw when attitude was logged
    (Tacview then draws the aircraft's orientation, not just its track)."""
    lat, lon, alt = _lla(pos)
    t = f"{lon:.7f}|{lat:.7f}|{alt:.1f}"
    if att is not None:
        roll, pitch, yaw = att
        t += f"|{roll:.1f}|{pitch:.1f}|{yaw % 360:.1f}"
    return t


def write_acmi(trajectory, path):
    if not trajectory:
        return
    with open(path, "w") as f:
        f.write("FileType=text/acmi/tacview\nFileVersion=2.1\n")
        f.write(f"0,ReferenceTime=2026-01-01T00:00:00Z\n")
        for row in trajectory:
            t = row["t"]
            f.write(f"#{t:.2f}\n")
            f.write(f"SELF,T={_transform(row['self_pos'], row.get('self_att'))},Name=F-16,Color=Blue\n")
            f.write(f"ADV,T={_transform(row['adv_pos'], row.get('adv_att'))},Name=F-16,Color=Red\n")
