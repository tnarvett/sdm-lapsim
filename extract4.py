import numpy as np
from PIL import Image
from scipy import ndimage
import sys

AX = r'C:\Users\tnarv\Downloads\autocross2026301c6283-1a76-4254-a1e8-3c10e248576f-md.png'
EN = r'C:\Users\tnarv\Downloads\enduro2026e0a16a9a-bfab-430e-8297-5af5ca610408-md.png'
load = lambda f: np.array(Image.open(f).convert('RGB')).astype(int)


def thin(mask):
    """vectorised Zhang-Suen."""
    I = mask.astype(np.uint8)
    def nb(a):
        p = [np.roll(np.roll(a, dy, 0), dx, 1)
             for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]]
        return p
    for _ in range(200):
        removed_any = False
        for step in (0, 1):
            P = nb(I)
            Bsum = sum(P)
            trans = np.zeros_like(I)
            for i in range(8):
                trans += ((P[i] == 0) & (P[(i + 1) % 8] == 1)).astype(np.uint8)
            cond = (I == 1) & (Bsum >= 2) & (Bsum <= 6) & (trans == 1)
            if step == 0:
                cond &= (P[0] * P[2] * P[4] == 0) & (P[2] * P[4] * P[6] == 0)
            else:
                cond &= (P[0] * P[2] * P[6] == 0) & (P[0] * P[4] * P[6] == 0)
            if cond.any():
                I[cond] = 0
                removed_any = True
        if not removed_any:
            break
    I[0, :] = I[-1, :] = I[:, 0] = I[:, -1] = 0
    return I.astype(bool)


def largest_cc(mask, n=1):
    lab, k = ndimage.label(mask, structure=np.ones((3, 3)))
    if k == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, k + 1))
    keep = np.argsort(sizes)[::-1][:n] + 1
    return np.isin(lab, keep)


def walk(sk, seed):
    pts = set(zip(*np.where(sk)[::-1]))
    path = [seed]; pts.discard(seed); cur = seed; prev = None
    for _ in range(8000):
        cx, cy = cur
        cands = [(cx+dx, cy+dy) for dx in (-1,0,1) for dy in (-1,0,1)
                 if (dx or dy) and (cx+dx, cy+dy) in pts]
        if not cands:
            best, bd = None, 1e9
            for dd in range(2, 18):
                for dx in range(-dd, dd+1):
                    for dy in range(-dd, dd+1):
                        p = (cx+dx, cy+dy)
                        if p in pts:
                            d = dx*dx+dy*dy
                            if prev:
                                d -= 2*((cx-prev[0])*dx + (cy-prev[1])*dy)
                            if d < bd: bd, best = d, p
                if best: break
            if best is None: break
            cands = [best]
        if prev and len(cands) > 1:
            vx, vy = cx-prev[0], cy-prev[1]
            cands.sort(key=lambda p: -((p[0]-cx)*vx + (p[1]-cy)*vy))
        nxt = cands[0]
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                pts.discard((nxt[0]+dx, nxt[1]+dy))
        path.append(nxt); prev, cur = cur, nxt
    return np.array(path, float)


def resample(pts, n):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    d = np.r_[0, np.cumsum(seg)]
    s = np.linspace(0, d[-1], n)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])], d[-1]


def gridspacing(im, y0frac=0.15, y1frac=0.85):
    dark = (im.max(axis=2) < 140)
    H = dark.shape[0]
    col = dark[int(H*y0frac):int(H*y1frac)].sum(axis=0).astype(float)
    col /= col.max()
    xs = [x for x in range(3, len(col)-3) if col[x] > 0.3 and col[x] >= col[x-1] and col[x] >= col[x+1]]
    xs = [x for i, x in enumerate(xs) if i == 0 or x - xs[i-1] > 3]
    d = np.diff(xs)
    d = d[(d > 6)]
    if len(d) == 0: return None, xs
    return float(np.median(d)), xs


def run(name, src, colorfn, seedfn, extra_close=3):
    im = load(src)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    m = colorfn(r, g, b)
    m = ndimage.binary_dilation(m, iterations=1)
    m = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=extra_close)
    m = largest_cc(m, 1)
    m = ndimage.binary_fill_holes(m)
    print(name, 'mask px', int(m.sum()))
    sk = largest_cc(thin(m), 1)
    print(name, 'skel px', int(sk.sum()))
    ys, xs = np.where(sk)
    seed = seedfn(xs, ys)
    p = walk(sk, seed)
    print(name, 'walk pts', len(p), 'span x', int(p[:,0].min()), int(p[:,0].max()), 'y', int(p[:,1].min()), int(p[:,1].max()))
    gsx, gxs = gridspacing(im)
    print(name, 'grid spacing px', gsx, ' first/last vline', (gxs[0], gxs[-1]) if gxs else None, 'n', len(gxs))
    ov = im.copy().astype(np.uint8)
    for x, y in p.astype(int):
        ov[max(0, y-1):y+2, max(0, x-1):x+2] = [255, 0, 255]
    Image.fromarray(ov).resize((ov.shape[1]*2, ov.shape[0]*2), Image.NEAREST).save(f'maps/{name}_walk.png')
    np.save(f'{name}_pathpx.npy', p)
    return p, im, gsx


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if w in ('ax', 'both'):
        run('ax', AX,
            lambda r, g, b: (r > 150) & (g < 115) & (b < 115) & (r - g > 55),
            lambda xs, ys: (int(xs.min()), int(np.median(ys[xs < xs.min() + 3]))),
            extra_close=2)
    if w in ('en', 'both'):
        # seed near bottom-right hairpin (max of x + weight for bottom)
        run('en', EN,
            lambda r, g, b: (r > 188) & (g > 100) & (g < 224) & (b > 100) & (b < 224) & (r - g > 15) & (r - b > 9),
            lambda xs, ys: tuple(int(v) for v in np.array([xs, ys]).T[np.argmax(xs.astype(float) + (ys > ys.mean()) * 500)]),
            extra_close=4)
