"""
Record a captioned walkthrough video of the demo web app.

Starts demo/app.py on a spare port with a clean, temporary run history; opens it in headless
Chrome (or Edge) at 1920x1080; injects tour.js, which drives the real page (pilots, start, two
runs, results, report) with a cursor, a spotlight and captions; captures every rendered frame
over the DevTools protocol (Page.startScreencast); and encodes them with ffmpeg into an H.264 MP4.
Nothing appears on the desktop.

  python demo/video/record_walkthrough.py                     -> videos/meol_demo_walkthrough.mp4
  python demo/video/record_walkthrough.py --out my.mp4 --keep-frames
"""

import argparse
import base64
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import websocket

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(os.path.dirname(HERE))
W, H = 1920, 1080           # video size in pixels
SCALE = 4 / 3               # device scale: the page lays out at 1440x810 CSS px and renders 1.33x, so it fills the frame
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


class CDP:
    """Minimal DevTools-protocol client: send() waits for its reply; events go to a queue."""

    def __init__(self, url):
        self.ws = websocket.create_connection(url, max_size=None, suppress_origin=True)
        self.ids, self.replies, self.events = itertools.count(1), {}, queue.Queue()
        self.cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                msg = json.loads(self.ws.recv())
            except Exception:
                self.events.put({"method": "__closed__"})
                return
            if "id" in msg:
                with self.cv:
                    self.replies[msg["id"]] = msg
                    self.cv.notify_all()
            else:
                self.events.put(msg)

    def send(self, method, params=None, wait=True):
        i = next(self.ids)
        self.ws.send(json.dumps({"id": i, "method": method, "params": params or {}}))
        if not wait:
            return None
        with self.cv:
            self.cv.wait_for(lambda: i in self.replies, timeout=30)
            return self.replies.pop(i, None)


def wait_http(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return urllib.request.urlopen(url, timeout=2).read()
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"{url} did not answer within {timeout} s")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default=os.path.join(RL, "videos", "meol_demo_walkthrough.mp4"))
    p.add_argument("--port", type=int, default=5055)
    p.add_argument("--debug-port", type=int, default=9333)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--timeout", type=int, default=1200, help="seconds before giving up on the tour")
    a = p.parse_args()

    work = tempfile.mkdtemp(prefix="meol_walkthrough_")
    runs, frames, profile = (os.path.join(work, d) for d in ("runs", "frames", "profile"))
    for d in (runs, frames, profile):
        os.makedirs(d)
    env = dict(os.environ, DEMO_PORT=str(a.port), DEMO_RUNS=runs)
    app = subprocess.Popen([sys.executable, os.path.join(RL, "demo", "app.py"), "--no-browser"], cwd=RL, env=env,
                           stdout=open(os.path.join(work, "app.log"), "w"), stderr=subprocess.STDOUT)
    browser = None
    try:
        wait_http(f"http://127.0.0.1:{a.port}/")
        exe = next(b for b in BROWSERS if os.path.exists(b))
        browser = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={a.debug_port}",
                                    f"--remote-allow-origins=http://127.0.0.1:{a.debug_port}",
                                    f"--user-data-dir={profile}", f"--window-size={W},{H}", "--hide-scrollbars",
                                    "--force-device-scale-factor=1", "--no-first-run", "--no-default-browser-check",
                                    "--disable-extensions", "about:blank"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        targets = json.loads(wait_http(f"http://127.0.0.1:{a.debug_port}/json"))
        page = next(t for t in targets if t["type"] == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.send("Emulation.setDeviceMetricsOverride", {"width": round(W / SCALE), "height": round(H / SCALE),
                                                         "deviceScaleFactor": SCALE, "mobile": False})
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": open(os.path.join(HERE, "tour.js"), encoding="utf-8").read()})
        cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})
        cdp.send("Page.navigate", {"url": f"http://127.0.0.1:{a.port}/"})

        stamps, t0, done = [], time.time(), False
        while not done and time.time() - t0 < a.timeout:
            try:
                ev = cdp.events.get(timeout=1)
            except queue.Empty:
                continue
            m = ev.get("method")
            if m == "Page.screencastFrame":
                prm = ev["params"]
                cdp.send("Page.screencastFrameAck", {"sessionId": prm["sessionId"]}, wait=False)
                path = os.path.join(frames, f"{len(stamps):06d}.jpg")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(prm["data"]))
                stamps.append((prm["metadata"]["timestamp"], path))
            elif m == "Runtime.consoleAPICalled":
                text = " ".join(str(x.get("value", "")) for x in ev["params"]["args"])
                if text.startswith("[tour]"):
                    print(f"{time.time() - t0:6.1f}s {text}  ({len(stamps)} frames)", flush=True)
                    done = "TOUR_DONE" in text
            elif m == "__closed__":
                break
        time.sleep(1.0)
        cdp.send("Page.stopScreencast", wait=False)
        if not done:
            raise SystemExit(f"the tour did not finish (see {work}\\app.log)")

        # constant-rate video from the frames' own timestamps (frames only arrive when the page changes)
        lst = os.path.join(work, "frames.txt")
        with open(lst, "w") as f:
            for (t, path), nxt in zip(stamps, stamps[1:] + [(stamps[-1][0] + 1.5, None)]):
                f.write(f"file '{path.replace(os.sep, '/')}'\nduration {max(nxt[0] - t, 0.001):.4f}\n")
            f.write(f"file '{stamps[-1][1].replace(os.sep, '/')}'\n")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                        "-vf", f"fps=30,scale={W}:{H}:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "slow",
                        "-crf", "20", "-tune", "stillimage", "-movflags", "+faststart", a.out], check=True)
        length = stamps[-1][0] - stamps[0][0] + 1.5
        print(f"wrote {a.out}: {length / 60:.1f} min, {len(stamps)} captured frames, "
              f"{os.path.getsize(a.out) / 1e6:.1f} MB")
    finally:
        if browser:
            browser.terminate()
        app.terminate()
        time.sleep(1.0)
        if not a.keep_frames:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print("frames kept in", work)


if __name__ == "__main__":
    main()
