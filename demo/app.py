"""
MEOL demo: a local web page to fly any two of the five pilots against each other.

  python demo/app.py            -> opens http://127.0.0.1:5000

Pick the own aircraft and the adversary (angle, snapshot, energy, MEOL option layer, rule-based
expert), the start (Range, AA, ATA) and the number of iterations. Each run writes a folder under
demo_runs/ with every fight's 3D plot, flight-log panels and Tacview file plus report.md, and the
page shows the results. Runs queue up and execute one at a time; fights within a run go in parallel.
"""

import json
import os
import queue
import sys
import threading
import traceback
import uuid
import webbrowser

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine  # noqa: E402

RUNS = os.environ.get("DEMO_RUNS") or os.path.join(engine.RL, "demo_runs")   # DEMO_RUNS: e.g. a clean folder for a recording
PILOT_INFO = {
    "angle": ("Nose-pointing advantage from the bandit's six (Eq. 38).", "23 W / 12 D / 3 L"),
    "snapshot": ("Aggressive: takes the shooting window at any aspect (Eq. 40, range-weighted AA).", "32 W / 5 D / 1 L"),
    "energy": ("Converts energy into position: nose-on reward scaled by the energy ratio (Eq. 41).", "30 W / 0 D / 8 L"),
    "meol": ("High level: picks angle, snapshot or energy every second (Algorithm 3).", "30 W / 8 D / 0 L"),
    "bfm": ("Shaw-doctrine rule-based expert, the opponent every policy was trained against.", "—"),
    "lag": ("HHMARL 3D's frozen network: LAG's F-16 autopilot flying HHMARL's pursuit logic, nose on the bandit "
            "and matching its altitude and speed.", "20 W / 9 D / 9 L"),
}


def _unavailable():
    """Pilots whose weights are not on disk (the third-party LAG network is fetched separately)."""
    return {k for k, p in engine.PILOTS.items() if p["path"] and not os.path.exists(os.path.join(engine.RL, p["path"]))}
PRESETS = [("Tail chase", 3000, 0, 0), ("Offensive, angle-off", 3000, 30, 20), ("Head-on", 4000, 180, 0),
           ("Beam, neutral", 3000, 90, 90), ("Crossing", 3500, 135, 45), ("Lateral split", 3500, 90, -90),
           ("Defensive", 2500, 180, 180), ("Defensive, offset", 3000, 150, 150)]

app = Flask(__name__)
JOBS, QUEUE = {}, queue.Queue()


def _worker():
    while True:
        job_id = QUEUE.get()
        job = JOBS[job_id]
        job["status"] = "running"

        def progress(done, total, r):
            job["done"] = done
            job["outcomes"][r["outcome"]] = job["outcomes"].get(r["outcome"], 0) + 1
            job["last"] = f"iteration {r['n']}: {r['outcome']} after {r['seconds']:.1f} s"

        try:
            job["run"] = engine.run_demo(job["cfg"], progress=progress)["run_name"]
            job["status"] = "done"
        except Exception:
            job["status"], job["error"] = "error", traceback.format_exc()


def _run_dir(run):
    if os.sep in run or "/" in run or run.startswith(".") or not os.path.isdir(os.path.join(RUNS, run)):
        abort(404)
    return os.path.join(RUNS, run)


def _recent(limit=15):
    out = []
    if os.path.isdir(RUNS):
        for name in sorted(os.listdir(RUNS), reverse=True):
            p = os.path.join(RUNS, name, "summary.json")
            if os.path.exists(p):
                out.append(json.load(open(p))["summary"])
            if len(out) >= limit:
                break
    return out


@app.get("/")
def index():
    q = request.args
    form = {"own": q.get("own", "meol"), "adversary": q.get("adversary", "bfm"), "range_m": q.get("range_m", 3000),
            "aa_deg": q.get("aa_deg", 0), "ata_deg": q.get("ata_deg", 0), "iterations": q.get("iterations", 10)}
    return render_template("index.html", pilots=engine.PILOTS, info=PILOT_INFO, presets=PRESETS, form=form,
                           recent=_recent(), queued=sum(j["status"] in ("queued", "running") for j in JOBS.values()),
                           unavailable=_unavailable())


@app.post("/run")
def run():
    f = request.form
    try:
        cfg = {"own": f["own"], "adversary": f["adversary"],
               "range_m": float(f["range_m"]), "aa_deg": float(f["aa_deg"]), "ata_deg": float(f["ata_deg"]),
               "iterations": int(f["iterations"]), "stochastic": "stochastic" in f, "weapons": "weapons" in f,
               "out_root": RUNS}
        if f.get("seed", "").strip():
            cfg["seed"] = int(f["seed"])
    except (KeyError, ValueError):
        abort(400, "bad form values")
    if cfg["own"] not in engine.PILOTS or cfg["adversary"] not in engine.PILOTS:
        abort(400, "unknown pilot")
    if {cfg["own"], cfg["adversary"]} & _unavailable():
        abort(400, "that pilot's weights are missing (LAG: python demo/lag_baseline.py --fetch)")
    if not (500 <= cfg["range_m"] <= 10000 and -180 <= cfg["aa_deg"] <= 180 and -180 <= cfg["ata_deg"] <= 180
            and 1 <= cfg["iterations"] <= 500):
        abort(400, "values out of range")
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"cfg": cfg, "status": "queued", "done": 0, "total": cfg["iterations"], "outcomes": {},
                    "last": "", "run": None, "error": None}
    QUEUE.put(job_id)
    return redirect(url_for("job", job_id=job_id))


@app.get("/job/<job_id>")
def job(job_id):
    if job_id not in JOBS:
        abort(404)
    j = JOBS[job_id]
    return render_template("job.html", job_id=job_id, cfg=j["cfg"], pilots=engine.PILOTS)


@app.get("/api/job/<job_id>")
def job_status(job_id):
    if job_id not in JOBS:
        abort(404)
    j = JOBS[job_id]
    earlier = list(JOBS)[:list(JOBS).index(job_id)]      # dicts keep submission order
    ahead = sum(JOBS[k]["status"] in ("queued", "running") for k in earlier) if j["status"] == "queued" else 0
    return jsonify({k: j[k] for k in ("status", "done", "total", "outcomes", "last", "run", "error")} | {"ahead": ahead})


@app.get("/runs/<run>")
def result(run):
    d = _run_dir(run)
    data = json.load(open(os.path.join(d, "summary.json")))
    its = data["iterations"]
    for r in its:
        r["has_options"] = os.path.exists(os.path.join(d, f"{r['files']}_options.png"))
    return render_template("result.html", s=data["summary"], its=its, run=run, folder=d)


@app.get("/runs/<run>/report")
def report(run):
    import markdown
    d = _run_dir(run)
    html = markdown.markdown(open(os.path.join(d, "report.md"), encoding="utf-8").read(), extensions=["tables"])
    return render_template("report.html", body=html, run=run)


@app.get("/files/<run>/<path:name>")
def files(run, name):
    return send_from_directory(_run_dir(run), name, as_attachment=name.endswith(".acmi"))


@app.post("/runs/<run>/open")
def open_folder(run):
    d = _run_dir(run)
    if hasattr(os, "startfile"):
        os.startfile(d)          # local app: open the run folder in Explorer
    return redirect(url_for("result", run=run))


def _free_port(start):
    """First port from `start` that nothing is bound to. The exclusive bind matters on Windows:
    there a second Flask server can bind a port already in use, and the two then split requests."""
    import socket
    for port in range(start, start + 50):
        s = socket.socket()
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            s.close()
    raise SystemExit(f"no free port in {start}-{start + 49}")


if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    wanted = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else int(os.environ.get("DEMO_PORT", 5000))
    port = _free_port(wanted)
    if port != wanted:
        print(f"port {wanted} is in use (another demo running?), using {port}", flush=True)
    print(f"MEOL demo on http://127.0.0.1:{port}", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
