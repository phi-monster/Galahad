"""c7_tax.py — measure single-3090 inference latency/Hz of the flagship via the eval socket. Client-side timing
(localhost RTT negligible). Reports amortized effective control Hz (N acts / total, over chunk boundaries) + the
cold first-act (chunk compile+compute) + per-act p50/max. Run with GALAHAD_FS_INFER=1 (foresight ON) vs 0 (OFF)."""
import os, socket, struct, pickle, time, statistics
import numpy as np
PORT = int(os.environ.get("PORT", "5000")); N = int(os.environ.get("N", "60")); TAG = os.environ.get("TAG", "x")


def _r(c, n):
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d:
            return None
        b += d
    return b


def rm(c):
    h = _r(c, 4); return None if h is None else pickle.loads(_r(c, struct.unpack(">I", h)[0]))


def sm(c, o):
    d = pickle.dumps(o, protocol=4); c.sendall(struct.pack(">I", len(d)) + d)


img = (np.zeros((256, 256, 3), np.uint8) + 120); wr = img.copy()
ob = {"image_b": img.tobytes(), "image_shape": [256, 256, 3], "image_dtype": "uint8",
      "wrist_b": wr.tobytes(), "wrist_shape": [256, 256, 3], "wrist_dtype": "uint8",
      "instruction": "pick up the alphabet soup and place it in the basket", "state": [0.0] * 8, "gt_pointer": [0.0, 0.0, 0.0]}
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM); conn.connect(("127.0.0.1", PORT))
sm(conn, {"cmd": "ping"}); assert rm(conn).get("ok")
sm(conn, {"cmd": "reset"}); rm(conn)
lat = []
t_all = time.time()
for i in range(N):
    t0 = time.time(); sm(conn, {"cmd": "act", "blank": False, "obs": ob}); rm(conn); lat.append(time.time() - t0)
total = time.time() - t_all
cold = lat[0]; warm = lat[1:]
amort_hz = N / total
print("C7[%s] N=%d  amortized_effective_Hz=%.2f (%.1f ms/step)  cold_first_act=%.3fs  warm_p50=%.4fs warm_max=%.4fs warm_min=%.4fs"
      % (TAG, N, amort_hz, 1000 * total / N, cold, statistics.median(warm), max(warm), min(warm)), flush=True)
print("C7_TAX_DONE_%s" % TAG, flush=True)
