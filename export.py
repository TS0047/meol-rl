"""Writes a Tacview ACMI 2.2 (text) file from a logged trajectory, optionally with a WEZ layer.

Format rules that matter (Tacview's ACMI spec): object ids are hexadecimal numbers, text is
UTF-8 with LF line endings, and T= is Longitude|Latitude|Altitude[|Roll|Pitch|Yaw] in
degrees and metres MSL, with yaw clockwise from true north.

WEZ layer (weapons=True, the default; WEAPONS below):
  * gun cone   each jet carries a wireframe cone of its weapon engagement zone -- the kill
               rule's +/-SHOOT_ATA_DEG from SHOOT_MIN_RANGE_M to SHOOT_RANGE_M -- as a child
               object drawn with the custom mesh WEZ_MESH (Tacview loads it from
               %APPDATA%\\Tacview\\Data\\Meshes; `python export.py --install-mesh` puts it there)
  * lock line  while a jet holds a one-sided gun solution -- env._check_outcome's rule -- it
               locks the other (LockedTarget), and Tacview draws the line
  * outcome    a timeline bookmark: WIN / LOSS / CRASH / TIMEOUT / DISENGAGED
No shooting or explosion effects.

  python export.py --install-mesh                         put the cone mesh where Tacview finds it
  python export.py --weapons on visualizations models     add the layer to existing files in place
  python export.py --weapons off visualizations           strip it again
  python export.py --upgrade logs runs                    convert files from the pre-fix writer
"""

import argparse
import math
import os
import re

import numpy as np

from env import BEHIT_AA_DEG, SHOOT_ATA_DEG, SHOOT_MIN_RANGE_M, SHOOT_RANGE_M   # the kill rule's WEZ

# The sim is flat-earth ENU metres around an origin; Tacview wants lon/lat. Place the origin at
# 0N 0E: open ocean at sea level, like JSBSim's default terrain, and degrees are nearly metric there.
REF_LAT, REF_LON = 0.0, 0.0
M_PER_DEG_LAT = 110574.0      # WGS-84 at the equator
M_PER_DEG_LON = 111320.0

SELF_ID, ADV_ID = "101", "102"                      # hex ids, as the spec requires
STATIC_PROPS = {
    SELF_ID: "Type=Air+FixedWing,Name=F-16C,Pilot={agent},Coalition=Allies,Color=Blue",
    ADV_ID: "Type=Air+FixedWing,Name=F-16C,Pilot={opponent},Coalition=Enemies,Color=Red",
}

WEAPONS = True                 # default for every writer; weapons=False / --no-weapons leaves the layer out
WEZ_MESH = "MEOL.WEZ.obj"      # file name Tacview looks up in Data\Meshes (case sensitive)
WEZ_MESH_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tacview", WEZ_MESH)
CONE_IDS = {SELF_ID: "111", ADV_ID: "112"}
CONE_EVERY_FRAMES = 5          # cone pose every 5th 50 Hz frame (10 Hz); Tacview interpolates between
OUTCOME_RE = re.compile(r"_(adversary_crash|win|loss|timeout|crash|disengaged)(?:_vs_bfm)?\.acmi$")


def _text(value):
    """Commas separate properties in ACMI, so a comma inside a text value is escaped."""
    return str(value).replace("\\", "\\\\").replace(",", "\\,").replace("\n", " ")


def _untext(value):
    return value.replace("\\,", ",").replace("\\\\", "\\")


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


def _nose(att):
    """Unit nose vector in ENU from (roll, pitch, yaw) degrees -- env's body_x_enu."""
    pitch, yaw = math.radians(att[1]), math.radians(att[2])
    return np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])


def wez_states(trajectory):
    """Per row: (agent holds a one-sided gun solution, opponent holds one), by env._check_outcome's
    rule. Uses the env's logged Range/ATA/AA, or recomputes them from positions and attitudes."""
    out = []
    for row in trajectory:
        if "Range" in row:
            rng, ata, aa = row["Range"], row["ATA"], row["AA"]
        elif row.get("self_att") is not None and row.get("adv_att") is not None:
            los = np.subtract(row["adv_pos"], row["self_pos"])
            rng = float(np.linalg.norm(los))
            u = los / max(rng, 1e-9)
            ata = math.degrees(math.acos(float(np.clip(_nose(row["self_att"]) @ u, -1, 1))))
            aa = math.degrees(math.acos(float(np.clip(_nose(row["adv_att"]) @ u, -1, 1))))
        else:
            out.append((False, False))
            continue
        band = SHOOT_MIN_RANGE_M <= rng <= SHOOT_RANGE_M
        s, a = band and ata <= SHOOT_ATA_DEG, band and aa >= BEHIT_AA_DEG
        out.append((s and not a, a and not s))
    return out


def write_acmi(trajectory, path, title="MEOL tactic engagement", opponent="adversary", outcome=None,
               weapons=WEAPONS, agent="SAC agent"):
    if not trajectory:
        return
    static = {oid: props.format(opponent=_text(opponent), agent=_text(agent)) for oid, props in STATIC_PROPS.items()}
    if weapons:
        wez = wez_states(trajectory)
    sides = ((SELF_ID, ADV_ID, "self", 0, "Blue"), (ADV_ID, SELF_ID, "adv", 1, "Red"))
    locked = {SELF_ID: False, ADV_ID: False}
    last = len(trajectory) - 1
    # utf-8-sig: the BOM Tacview recommends; newline="\n": no CRLF on Windows
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("FileType=text/acmi/tacview\nFileVersion=2.2\n")
        f.write("0,ReferenceTime=2026-01-01T00:00:00Z\n")
        f.write(f"0,Title={_text(title)}\n0,DataSource=JSBSim F-16 (MEOL reproduction)\n")
        for i, row in enumerate(trajectory):
            f.write(f"#{row['t']:.2f}\n")
            for oid, other, key, k, color in sides:
                tr = _transform(row[f"{key}_pos"], row.get(f"{key}_att"))
                extra = "," + static[oid] if i == 0 else ""   # static properties once
                if weapons and wez[i][k] != locked[oid]:
                    locked[oid] = wez[i][k]
                    extra += f",LockedTargetMode=1,LockedTarget={other}" if locked[oid] else ",LockedTargetMode=0"
                f.write(f"{oid},T={tr}{extra}\n")
                if weapons and row.get(f"{key}_att") is not None and (i % CONE_EVERY_FRAMES == 0 or i == last):
                    cone = f",Type=Misc+Minor,Name=WEZ,Shape={WEZ_MESH},Parent={oid},Color={color}" if i == 0 else ""
                    f.write(f"{CONE_IDS[oid]},T={tr}{cone}\n")       # the cone rides on its jet
        if weapons and outcome:
            label = {"win": f"WIN: gun kill by {agent}",
                     "loss": f"LOSS: gun kill by {opponent}",
                     "crash": f"CRASH: {agent} hit the ground",
                     "adversary_crash": f"{opponent} hit the ground",
                     "timeout": f"TIMEOUT: draw after {trajectory[-1]['t']:.0f} s",
                     "disengaged": "DISENGAGED: range over 12 km"}.get(outcome, outcome)
            f.write(f"0,Event=Bookmark|{_text(label)}\n")


def wez_mesh_obj(half_angle=SHOOT_ATA_DEG, r_min=SHOOT_MIN_RANGE_M, r_max=SHOOT_RANGE_M, ribs=8, segments=48,
                 width=6.0):
    """Wavefront OBJ text of a wireframe WEZ cone in Tacview's mesh frame (-Z forward, Y up, metres):
    8 ribs along the cone from r_min to r_max and a ring at each range. Every bar is two crossed
    strips, each triangle in both windings, so the frame shows from any angle; Tacview ignores
    materials and normals and takes triangles only."""
    th = math.radians(half_angle)
    verts, faces = [], []

    def on_cone(r, phi):
        return np.array([r * math.sin(th) * math.cos(phi), r * math.sin(th) * math.sin(phi), -r * math.cos(th)])

    def bar(p0, p1, side):
        n = len(verts) + 1
        verts.extend([p0 - side, p0 + side, p1 + side, p1 - side])
        faces.extend([(n, n + 1, n + 2), (n, n + 2, n + 3), (n, n + 2, n + 1), (n, n + 3, n + 2)])

    for k in range(ribs):
        phi = 2 * math.pi * k / ribs
        p0, p1 = on_cone(r_min, phi), on_cone(r_max, phi)
        tangent, radial = np.array([-math.sin(phi), math.cos(phi), 0.0]), np.array([math.cos(phi), math.sin(phi), 0.0])
        bar(p0, p1, tangent * width / 2)
        bar(p0, p1, radial * width / 2)
    for r in (r_min, r_max):
        for j in range(segments):
            a, b = 2 * math.pi * j / segments, 2 * math.pi * (j + 1) / segments
            mid = (a + b) / 2
            bar(on_cone(r, a), on_cone(r, b), np.array([0.0, 0.0, width / 2]))
            bar(on_cone(r, a), on_cone(r, b), np.array([math.cos(mid), math.sin(mid), 0.0]) * width / 2)
    lines = [f"# WEZ cone: +/-{half_angle:g} deg, {r_min:g}-{r_max:g} m; -Z forward, Y up, metres ({len(faces)} triangles)"]
    lines += [f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}" for v in verts] + [f"f {a} {b} {c}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def install_mesh():
    """Write the cone mesh to tacview/ in the repo and copy it into Tacview's user mesh folder."""
    os.makedirs(os.path.dirname(WEZ_MESH_SRC), exist_ok=True)
    with open(WEZ_MESH_SRC, "w", newline="\n") as f:
        f.write(wez_mesh_obj())
    dst_dir = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "Tacview", "Data", "Meshes")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, WEZ_MESH)
    with open(WEZ_MESH_SRC) as src, open(dst, "w", newline="\n") as out:
        out.write(src.read())
    return dst


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


def read_acmi(path):
    """(rows, meta) from any file this module wrote, old or new format; meta holds title and opponent.
    Weapon objects and events are skipped, so a rewritten file never duplicates them."""
    if is_legacy(path):
        return read_legacy_acmi(path), {}
    rows, row, meta = [], None, {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                row = {"t": float(line[1:])}
                rows.append(row)
            elif line.startswith("0,Title="):
                meta["title"] = _untext(line[len("0,Title="):])
            elif line.startswith((SELF_ID + ",", ADV_ID + ",")):
                oid, rest = line.split(",", 1)
                key = "self" if oid == SELF_ID else "adv"
                for prop in re.split(r"(?<!\\),", rest):
                    if prop.startswith("T="):
                        v = [float(x) for x in prop[2:].split("|")]
                        row[f"{key}_pos"] = ((v[0] - REF_LON) * M_PER_DEG_LON, (v[1] - REF_LAT) * M_PER_DEG_LAT, v[2])
                        row[f"{key}_att"] = tuple(v[3:6]) if len(v) >= 6 else None
                    elif prop.startswith("Pilot="):
                        meta["opponent" if oid == ADV_ID else "agent"] = _untext(prop[len("Pilot="):])
    return [r for r in rows if "self_pos" in r and "adv_pos" in r], meta


def is_legacy(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return any(line.startswith(("SELF,", "ADV,")) for _, line in zip(range(8), f))


def outcome_from_name(path):
    m = OUTCOME_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser(description="Rewrite existing ACMI files in place: convert old-format files, "
                                            "or add / remove the weapons layer.")
    p.add_argument("paths", nargs="*", help="files or folders (searched recursively)")
    p.add_argument("--upgrade", action="store_true", help="convert files from the pre-fix writer (ids SELF/ADV)")
    p.add_argument("--weapons", choices=["on", "off"], help="rewrite every file with or without the WEZ layer")
    p.add_argument("--install-mesh", action="store_true", help="install the WEZ cone mesh into Tacview's mesh folder")
    args = p.parse_args()
    if args.install_mesh:
        print("WEZ cone mesh installed:", install_mesh())
    if not (args.upgrade or args.weapons):
        if not args.install_mesh:
            p.error("pass --install-mesh, --upgrade and/or --weapons on|off")
        return
    files = []
    for target in args.paths:
        if os.path.isdir(target):
            files += [os.path.join(d, n) for d, _, ns in os.walk(target) for n in ns if n.endswith(".acmi")]
        else:
            files.append(target)
    done = 0
    for path in files:
        if args.weapons is None and not is_legacy(path):
            continue
        rows, meta = read_acmi(path)
        write_acmi(rows, path, title=meta.get("title", os.path.splitext(os.path.basename(path))[0]),
                   opponent=meta.get("opponent", "adversary"), agent=meta.get("agent", "SAC agent"),
                   outcome=outcome_from_name(path),
                   weapons=(args.weapons == "on") if args.weapons else WEAPONS)
        done += 1
    print(f"rewrote {done} of {len(files)} ACMI files" + (f" with weapons {args.weapons}" if args.weapons else ""))


if __name__ == "__main__":
    main()
