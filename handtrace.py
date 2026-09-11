"""DEPRECATED (Sept 2026). Hand-traced FSAE Michigan 2026 endurance + autocross,
read off the map grid, coords in FEET (x from the map's right-hand 0', y up).

Superseded by extract_tracks.py, which auto-extracts the centrelines straight from
the map PNGs (colour-segment -> skeletonize -> graph the loop -> sub-pixel centre)
and — critically — uses the grid-locked scale (endurance 0.7679 m/px, autocross
0.6645 m/px). This hand-trace and the old EN_NODES/AX_NODES it fed were ~13 %
(endurance) / ~19 % (autocross) undersized. Kept only for reference.
"""
import numpy as np
from PIL import Image

FT = 0.3048

# ---- ENDURANCE (closed loop), travel: start -> right along top -> down right ->
#      BR hairpin -> left along serpentine -> up left side -> TL corner ->
#      top sweeper down -> dip -> esses -> start
EN_FT = [
    (1050, 574), (985, 588), (920, 566), (855, 585), (790, 562), (725, 586),
    (660, 566), (595, 588), (530, 566), (465, 585), (400, 570), (330, 586),
    (255, 572), (170, 588), (95, 574),
    (40, 545), (28, 500), (46, 442), (34, 390), (56, 340), (40, 292),
    (62, 250), (44, 208), (82, 172),
    # BR hairpin (wide U, ~180deg)
    (55, 148), (24, 150), (10, 178), (22, 208), (55, 218), (90, 200), (104, 150),
    (100, 108),
    # bottom serpentine (x increasing = travelling left)
    (160, 104), (240, 122), (300, 178), (362, 128), (432, 98), (520, 132),
    (600, 188), (680, 128), (742, 92), (802, 152), (862, 202), (922, 128),
    (982, 78), (1042, 122), (1102, 178), (1162, 128), (1222, 94),
    # serpentine hairpin (tight U ~x 1270-1385)
    (1266, 82), (1345, 68), (1388, 106), (1338, 136), (1280, 122),
    (1306, 150), (1362, 168), (1422, 108), (1472, 84), (1522, 132),
    (1562, 178), (1612, 120), (1672, 90), (1732, 132), (1782, 182),
    (1842, 130), (1892, 94), (1930, 112), (1956, 90), (1976, 122),
    # BL corner sweep up
    (2002, 162), (2018, 222), (2012, 292),
    # left side straight (y increasing) with gentle S
    (2018, 360), (1996, 432), (2022, 502), (2010, 556),
    # TL corner (sharp, small bump at apex)
    (1992, 602), (1956, 616), (1912, 602), (1872, 576),
    # top sweeper descending right (x decreasing, y decreasing)
    (1782, 560), (1682, 540), (1582, 510), (1492, 470), (1412, 426),
    # the dip (V valley, pointy)
    (1372, 400), (1345, 379), (1362, 402), (1392, 432),
    # esses climbing out (tight S-kinks)
    (1322, 460), (1290, 492), (1250, 480), (1230, 512), (1200, 500),
    (1170, 532), (1140, 520), (1110, 546), (1080, 556),
]

# ---- AUTOCROSS (open course), travel: start (green flag, upper lane) -> right
#      with 2 chicane offsets -> hairpin -> back left on lower lane -> finish
AX_FT = [
    (455, 146), (498, 160), (545, 134), (595, 162), (645, 132), (695, 158),
    (748, 172), (800, 134), (855, 160), (905, 138), (955, 162), (1000, 146),
    (1022, 108), (1070, 104), (1115, 110),           # chicane step down
    (1145, 152), (1188, 140), (1235, 126),           # chicane step up
    (1285, 172), (1328, 132), (1368, 118), (1418, 160), (1468, 188),
    (1518, 146), (1558, 124), (1590, 164),
    # hairpin (wide, 180deg)
    (1624, 178), (1660, 156), (1670, 108), (1652, 70), (1614, 50),
    # lower lane back (x decreasing)
    (1558, 58), (1508, 90), (1458, 56), (1408, 40), (1358, 72), (1308, 100),
    (1258, 60), (1208, 44), (1158, 78), (1128, 56), (1088, 74), (1048, 50),
    (1008, 80), (958, 56), (908, 42), (858, 76), (808, 50), (758, 70),
    (718, 46), (688, 72), (666, 50),
]


def to_nodes(ft, mirror_x=True):
    a = np.array(ft, float)
    if mirror_x:  # map x grows leftward; flip so model x grows rightward
        a[:, 0] = a[:, 0].max() - a[:, 0]
    a *= FT
    a -= a.min(axis=0)
    return a


def curvature(xy, closed):
    n = len(xy); k = np.zeros(n)
    for i in range(n):
        A, B, C = (i - 1) % n, i, (i + 1) % n
        if not closed and (i == 0 or i == n - 1):
            continue
        ax, ay = xy[A]; bx, by = xy[B]; cx, cy = xy[C]
        ar = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        dd = np.hypot(bx-ax, by-ay) * np.hypot(cx-bx, cy-by) * np.hypot(ax-cx, ay-cy)
        k[i] = 2 * ar / dd if dd > 1e-6 else 0
    return k


def report(name, ft, closed):
    xy = to_nodes(ft)
    seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]) if closed else xy, axis=0).T)
    L = seg.sum()
    k = np.abs(curvature(xy, closed))
    minR = 1 / k.max() if k.max() > 1e-4 else 9999
    print(f'{name}: {len(xy)} nodes  length {L:.0f} m ({L/FT:.0f} ft)  minR {minR:.1f} m')
    js = '[' + ','.join(f'[{x:.1f},{y:.1f}]' for x, y in xy) + ']'
    open(f'{name}_nodes_ht.txt', 'w').write(js)
    return xy


def overlay(name, srcimg, ft, mpx, closed):
    im = np.array(Image.open(srcimg).convert('RGB')).astype(np.uint8)
    a = np.array(ft, float)
    # back to px: x_px = (xmax - x)/? ... map x is from right. tick calib: 228px=500ft (EN), .554px/ft (AX)
    ppf = {'en': 228 / 500, 'ax': 0.554}[name]
    # need origin. EN: x=0 at right edge-ish. find rightmost tick.
    x0 = {'en': im.shape[1] - 30, 'ax': 56}[name]
    sgn = {'en': -1, 'ax': 1}[name]
    # y: rough - place using known feature. EN top ~ y_px 95 for y_ft ~600 ; bottom y_px ~ 430 for y_ft ~30
    if name == 'en':
        yf = np.polyfit([600, 30], [95, 430], 1)
    else:
        yf = np.polyfit([145, 5], [40, 95], 1)
    for X, Y in a:
        px = int(x0 + sgn * X * ppf)
        py = int(np.polyval(yf, Y))
        if 0 <= py < im.shape[0] and 0 <= px < im.shape[1]:
            im[max(0, py-2):py+3, max(0, px-2):px+3] = [255, 0, 0]
    Image.fromarray(im).resize((im.shape[1]*2, im.shape[0]*2), Image.NEAREST).save(f'maps/{name}_ht_overlay.png')


if __name__ == '__main__':
    report('en', EN_FT, True)
    report('ax', AX_FT, False)
    overlay('en', r'C:\Users\tnarv\Downloads\enduro2026e0a16a9a-bfab-430e-8297-5af5ca610408-md.png', EN_FT, None, True)
    overlay('ax', r'C:\Users\tnarv\Downloads\autocross2026301c6283-1a76-4254-a1e8-3c10e248576f-md.png', AX_FT, None, False)
