"""
Auto-extract the FSAE Michigan 2026 autocross + endurance centrelines from the
official course-map PNGs. Supersedes the hand-trace in handtrace.py.

Pipeline
--------
1. Colour-segment the drawn racing line
       autocross : bright red   (R>135, G<115, B<115, R-G>55, R-B>55)
       endurance : salmon/pink  (R>178, R-G>16, R-B>16, ...), pit-lane island masked out
2. Keep the largest connected component, skeletonize (skimage), and:
       autocross : walk the single open path green-flag -> chequered-flag
       endurance : build the skeleton graph (sknw), take the max-enclosed-area cycle
3. Sub-pixel refine each point: centroid of line intensity along the local normal.
4. Resample to 2 m, Savitzky-Golay smooth (win 9), local de-kink below R=4 m.
5. Resample to node spacing, convert px -> metres, flip Y, shift to +quadrant.

Scale  (locked to the printed grid / axis labels, NOT the old 0.6684 / 0.5502)
-------
   endurance  100 ft = 39.69 px  ->  0.76793 m/px   (26-gridline fit, resid 0.5 px)
   autocross  100 ft = 45.92 px  ->  0.66448 m/px   (17 axis-label fit,  resid 0.4 px)
The previous calibration was ~13 % (endurance) / ~19 % (autocross) undersized.

Outputs
-------
   EN_FINAL.txt / AX_FINAL.txt   -- JS node arrays, paste into EN_NODES / AX_NODES in lapsim.html
   maps/EN_overlay.png / maps/AX_overlay.png  -- trace drawn back over the source map for QA

Requires: numpy, scipy, scikit-image, sknw, networkx, pillow
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize, closing, disk, remove_small_objects
from skimage.measure import label
import sknw
import networkx as nx

MAPS = "maps"
EN_MPX = 0.76793
AX_MPX = 0.66448

# pit-lane / paddock island near the endurance start-finish (map px) -- masked so the
# max-area cycle follows the racing line, not the pit outline.
EN_PIT_BOX = (452, 632, 18, 45)   # x0, x1, y0, y1

# autocross skeleton endpoints (map px): green start flag, chequered finish flag
AX_START = (31, 343)
AX_FINISH = (70, 450)


def line_field(img):
    a = np.asarray(img).astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    return np.clip(R - np.maximum(G, B), 0, None) * ((R > 170) & (R - G > 14) & (R - B > 14))


def refine(P, field, win=2.5, ns=21, tan=3, closed=True):
    n = len(P)
    out = []
    for i in range(n):
        a_, b_ = (P[(i - tan) % n], P[(i + tan) % n]) if closed else (P[max(0, i - tan)], P[min(n - 1, i + tan)])
        dy, dx = b_[0] - a_[0], b_[1] - a_[1]
        L = np.hypot(dx, dy) + 1e-9
        nx_, ny_ = -dy / L, dx / L
        num, den = np.zeros(2), 0.0
        for t in np.linspace(-win, win, ns):
            yy, xx = P[i, 0] + ny_ * t, P[i, 1] + nx_ * t
            iy, ix = int(round(yy)), int(round(xx))
            if 0 <= iy < field.shape[0] and 0 <= ix < field.shape[1]:
                w = field[iy, ix]
                num += w * np.array([yy, xx])
                den += w
        out.append(num / den if den > 0 else P[i])
    return np.array(out)


def curv_R(x, y, closed):
    n = len(x)
    if closed:
        xp, yp = np.r_[x[-2:], x, x[:2]], np.r_[y[-2:], y, y[:2]]
    else:
        xp, yp = np.r_[x[0], x[0], x, x[-1], x[-1]], np.r_[y[0], y[0], y, y[-1], y[-1]]
    dx = ((xp[2:] - xp[:-2]) / 2)[:n]
    dy = ((yp[2:] - yp[:-2]) / 2)[:n]
    ddx = (xp[2:] - 2 * xp[1:-1] + xp[:-2])[:n]
    ddy = (yp[2:] - 2 * yp[1:-1] + yp[:-2])[:n]
    return 1 / np.maximum(np.abs(dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9), 1e-9)


def dekink(x, y, closed, rmin=4.0, passes=20):
    x, y = x.copy(), y.copy()
    n = len(x)
    for _ in range(passes):
        bad = np.where(curv_R(x, y, closed) < rmin)[0]
        if not len(bad):
            break
        for i in bad:
            a, b = ((i - 1) % n, (i + 1) % n) if closed else (max(0, i - 1), min(n - 1, i + 1))
            x[i] = 0.4 * x[i] + 0.3 * (x[a] + x[b])
            y[i] = 0.4 * y[i] + 0.3 * (y[a] + y[b])
    return x, y


def path_to_nodes(P_px, field, closed, mpx, node_m):
    keep = [0]
    for i in range(1, len(P_px)):
        if np.hypot(*(P_px[i] - P_px[keep[-1]])) > 0.6:
            keep.append(i)
    ref = refine(P_px[keep], field, closed=closed)
    xp, yp = ref[:, 1], ref[:, 0]
    xy = np.vstack([np.c_[xp, yp], [[xp[0], yp[0]]]]) if closed else np.c_[xp, yp]
    d = np.hypot(*np.diff(xy, axis=0).T)
    s = np.r_[0, np.cumsum(d)]
    su = np.arange(0, s[-1], 2.0)
    xf = np.interp(su, s, xy[:, 0])
    yf = np.interp(su, s, xy[:, 1])
    wl, p = 9, 9
    if closed:
        xb = savgol_filter(np.r_[xf[-p:], xf, xf[:p]], wl, 2)[p:-p]
        yb = savgol_filter(np.r_[yf[-p:], yf, yf[:p]], wl, 2)[p:-p]
    else:
        xb = savgol_filter(xf, wl, 2)
        yb = savgol_filter(yf, wl, 2)
    xb, yb = dekink(xb, yb, closed)
    dd = np.hypot(np.diff(np.r_[xb, xb[0]] if closed else xb), np.diff(np.r_[yb, yb[0]] if closed else yb))
    sc = np.r_[0, np.cumsum(dd)]
    Lm = sc[-1] * mpx
    nn = max(8, int(round(Lm / node_m)))
    sn = np.linspace(0, sc[-1], nn, endpoint=not closed)
    xnp = np.interp(sn, sc, np.r_[xb, xb[0]] if closed else xb)
    ynp = np.interp(sn, sc, np.r_[yb, yb[0]] if closed else yb)
    xn, yn = xnp * mpx, -ynp * mpx
    if closed:                              # start on the main straight, run east
        Rf = curv_R(xn, yn, True)
        cand = np.where(yn > yn.max() - 15)[0]
        s0 = cand[np.argmax(Rf[cand])]
        xn, yn = np.roll(xn, -s0), np.roll(yn, -s0)
        if xn[1] < xn[0]:
            xn = np.roll(xn[::-1], 1).copy()
            yn = np.roll(yn[::-1], 1).copy()
    xn = xn - xn.min() + 2
    yn = yn - yn.min() + 2
    return np.c_[np.round(xn, 1), np.round(yn, 1)], Lm, xnp, ynp


def extract_autocross():
    img = Image.open(f"{MAPS}/autocross_2026.png").convert("RGB")
    a = np.asarray(img).astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    red = (R > 135) & (G < 115) & (B < 115) & (R - G > 55) & (R - B > 55)
    lb = label(red)
    sz = np.bincount(lb.ravel()); sz[0] = 0
    course = lb == np.argmax(sz)
    sk = skeletonize(course)
    pts = set(map(tuple, np.argwhere(sk)))
    path = [AX_START]; pts.discard(AX_START); cur = AX_START
    while pts:
        y, x = cur; best = None; bd = 99
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    c = (y + dy, x + dx)
                    if c in pts and dy * dy + dx * dx < bd:
                        bd, best = dy * dy + dx * dx, c
        if best is None:
            break
        path.append(best); pts.discard(best); cur = best
        if best == AX_FINISH:
            break
    return path_to_nodes(np.array(path, float), line_field(img), False, AX_MPX, 11.0), img


def extract_endurance():
    img = Image.open(f"{MAPS}/endurance_2026.png").convert("RGB")
    a = np.asarray(img).astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    H, W = R.shape
    pink = (R > 178) & (R - G > 16) & (R - B > 16) & (G > 65) & (B > 65) & (np.abs(G - B) < 58)
    pink &= ~((R > 170) & (G < 78) & (B < 78))          # drop red/yellow flag markers
    pink = remove_small_objects(pink, 6)
    yy, xx = np.mgrid[0:H, 0:W]
    x0, x1, y0, y1 = EN_PIT_BOX
    pink &= ~((xx > x0) & (xx < x1) & (yy > y0) & (yy < y1))
    m = closing(pink, disk(4))
    m = remove_small_objects(m, 60)
    lb = label(m)
    sz = np.bincount(lb.ravel()); sz[0] = 0
    sk = skeletonize(lb == np.argmax(sz))
    Gg = sknw.build_sknw(sk, multi=True)
    H_ = Gg.copy()
    ch = True
    while ch:
        ch = False
        for n in list(H_.nodes()):
            if H_.degree(n) <= 1:
                H_.remove_node(n); ch = True
    SG = nx.Graph()
    for u, v, dd in H_.edges(data=True):
        L = len(dd["pts"])
        if not SG.has_edge(u, v) or L > SG[u][v]["w"]:
            SG.add_edge(u, v, w=L, pts=dd["pts"])
    best = None
    for cyc in nx.simple_cycles(SG):
        if len(cyc) < 3:
            continue
        segs = []
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            seg = np.array(SG.get_edge_data(u, v)["pts"], float)
            nu = np.array(H_.nodes[u]["o"])
            if np.linalg.norm(seg[0] - nu) > np.linalg.norm(seg[-1] - nu):
                seg = seg[::-1]
            segs.append(seg)
        P = np.vstack(segs)
        area = 0.5 * abs(np.dot(P[:, 1], np.roll(P[:, 0], 1)) - np.dot(P[:, 0], np.roll(P[:, 1], 1)))
        if best is None or area > best[0]:
            best = (area, P)
    return path_to_nodes(best[1], line_field(img), True, EN_MPX, 17.0), img


def overlay(img, xnp, ynp, closed, path):
    S = 5
    base = (np.asarray(img.convert("L").convert("RGB")) * 0.45 + 140).astype("uint8")
    c = Image.fromarray(base).resize((img.width * S, img.height * S), Image.NEAREST)
    d = ImageDraw.Draw(c)
    pl = [(float(xnp[i] * S), float(ynp[i] * S)) for i in range(len(xnp))]
    if closed:
        pl.append(pl[0])
    d.line(pl, fill=(0, 90, 235), width=4)
    if not closed:
        d.ellipse([xnp[0] * S - 11, ynp[0] * S - 11, xnp[0] * S + 11, ynp[0] * S + 11], outline=(0, 190, 0), width=5)
        d.ellipse([xnp[-1] * S - 11, ynp[-1] * S - 11, xnp[-1] * S + 11, ynp[-1] * S + 11], outline=(230, 0, 0), width=5)
    c.save(path)


def js(arr):
    return "[" + ",".join(f"[{x:.1f},{y:.1f}]" for x, y in arr) + "]"


if __name__ == "__main__":
    (AX, axL, axx, axy), aximg = extract_autocross()
    (EN, enL, enx, eny), enimg = extract_endurance()
    open("AX_FINAL.txt", "w").write(js(AX) + "\n")
    open("EN_FINAL.txt", "w").write(js(EN) + "\n")
    overlay(aximg, axx, axy, False, f"{MAPS}/AX_overlay.png")
    overlay(enimg, enx, eny, True, f"{MAPS}/EN_overlay.png")
    print(f"autocross : {len(AX):3d} nodes, ~{axL:.0f} m  (open)")
    print(f"endurance : {len(EN):3d} nodes, ~{enL:.0f} m  (closed)")
    print("wrote AX_FINAL.txt / EN_FINAL.txt + maps/*_overlay.png")
    print("-> paste the arrays into AX_NODES / EN_NODES in lapsim.html")
