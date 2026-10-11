"""Writes a Tacview ACMI 2.2 (text) file from a logged trajectory.

Format rules that matter (Tacview's ACMI spec): object ids are hexadecimal numbers, text is
UTF-8 with LF line endings, and T= is Longitude|Latitude|Altitude[|Roll|Pitch|Yaw] in
degrees and metres MSL, with yaw clockwise from true north.

  python export.py --upgrade logs runs visualizations    rewrite files from the old writer in place
"""

import argparse
import os

# The sim is flat-earth ENU metres around an origin; Tacview wants lon/lat. Place the origin at
# 0N 0E: open ocean at sea level, like JSBSim's default terrain, and degrees are nearly metric there.
REF_LAT, REF_LON = 0.0, 0.0
M_PER_DEG_LAT = 110574.0      # WGS-84 at the equator
M_PER_DEG_LON = 111320.0

SELF_ID, ADV_ID = "101", "102"                      # hex ids, as the spec requires
STATIC_PROPS = {
    SELF_ID: "Type=Air+FixedWing,Name=F-16C,Pilot=SAC agent,Coalition=Allies,Color=Blue",
    ADV_ID: "Type=Air+FixedWing,Name=F-16C,Pilot={opponent},Coalition=Enemies,Color=Red",
}


def _text(value):
    """Commas separate properties in ACMI, so a comma inside a text value is escaped."""
    return str(value).replace("\\", "\\\\").replace(",", "\\,").replace("\n", " ")


def _lonlat(pos_enu):
    return REF_LON + pos_enu[0] / M_PER_DEG_LON, REF_LAT + pos_enu[1] / M_PER_DEG_LAT


def _transform(pos, att):
    """ACMI T= field: lon|lat|alt, plus |roll|pitch|yaw when attitude was logged
    (Tacview then draws the aircraft's orientation, not just its track)."""
    lon, lat = _lonlat(pos)
    t = f"{lon:.7f}|{lat:.7f}|{pos[2]:.1f}"
    if att is not None:
        roll, pitch, yaw = att
        t += f"|{roll:.1f}|{pitch:.1f}|{yaw % 360:.1f}"
    return t


def write_acmi(trajectory, path, title="MEOL angle tactic engagement", opponent="adversary"):
    if not trajectory:
        return
    static = {oid: props.format(opponent=_text(opponent)) for oid, props in STATIC_PROPS.items()}
    # utf-8-sig: the BOM Tacview recommends; newline="\n": no CRLF on Windows
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("FileType=text/acmi/tacview\nFileVersion=2.2\n")
        f.write("0,ReferenceTime=2026-01-01T00:00:00Z\n")
        f.write(f"0,Title={_text(title)}\n0,DataSource=JSBSim F-16 (MEOL reproduction)\n")
        for i, row in enumerate(trajectory):
            f.write(f"#{row['t']:.2f}\n")
            for oid, pos, att in ((SELF_ID, row["self_pos"], row.get("self_att")),
                                  (ADV_ID, row["adv_pos"], row.get("adv_att"))):
                extra = "," + static[oid] if i == 0 else ""   # static properties once
                f.write(f"{oid},T={_transform(pos, att)}{extra}\n")


def read_legacy_acmi(path):
    """Trajectory rows from a file of the old writer (ids SELF/ADV, which Tacview rejects).
    That writer used 1/111320 deg per metre on both axes, so this undoes exactly that."""
    old = 111320.0
    rows, row = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                row = {"t": float(line[1:])}
                rows.append(row)
            elif line.startswith(("SELF,T=", "ADV,T=")):
                oid, rest = line.split(",T=", 1)
                v = [float(x) for x in rest.split(",", 1)[0].split("|")]
                key = "self" if oid == "SELF" else "adv"
                row[f"{key}_pos"] = (v[0] * old, v[1] * old, v[2])
                row[f"{key}_att"] = tuple(v[3:6]) if len(v) >= 6 else None
    return rows


def is_legacy(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return any(line.startswith(("SELF,", "ADV,")) for _, line in zip(range(8), f))


def main():
    p = argparse.ArgumentParser(description="Rewrite old-format ACMI files so Tacview can open them.")
    p.add_argument("--upgrade", nargs="+", metavar="PATH", required=True,
                   help="files or folders (searched recursively) to upgrade in place")
    args = p.parse_args()
    files = []
    for target in args.upgrade:
        if os.path.isdir(target):
            files += [os.path.join(d, n) for d, _, ns in os.walk(target) for n in ns if n.endswith(".acmi")]
        else:
            files.append(target)
    done = 0
    for path in files:
        if is_legacy(path):
            write_acmi(read_legacy_acmi(path), path, title=os.path.splitext(os.path.basename(path))[0])
            done += 1
    print(f"upgraded {done} of {len(files)} ACMI files (the rest were already in the new format)")


if __name__ == "__main__":
    main()
