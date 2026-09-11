import numpy as np
from PIL import Image
from scipy import ndimage
import sys, json

AX = r'C:\Users\tnarv\Downloads\autocross2026301c6283-1a76-4254-a1e8-3c10e248576f-md.png'
EN = r'C:\Users\tnarv\Downloads\enduro2026e0a16a9a-bfab-430e-8297-5af5ca610408-md.png'
MPX = {'ax': 0.5507, 'en': 0.6684}
load = lambda f: np.array(Image.open(f).convert('RGB')).astype(int)

def thin(mask):
    I = mask.astype(np.uint8)
    def nb(a):
        return [np.roll(np.roll(a, dy, 0), dx, 1) for dy,dx in
                [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]]
    for _ in range(300):
        any_removed = False
        for step in (0,1):
            P = nb(I); B = sum(P)
            T = np.zeros_like(I)
            for i in range(8):
                T += ((P[i]==0)&(P[(i+1)%8]==1)).astype(np.uint8)
            c = (I==1)&(B>=2)&(B<=6)&(T==1)
            if step==0: c &= (P[0]*P[2]*P[4]==0)&(P[2]*P[4]*P[6]==0)
            else:       c &= (P[0]*P[2]*P[6]==0)&(P[0]*P[4]*P[6]==0)
            if c.any(): I[c]=0; any_removed=True
        if not any_removed: break
    I[0,:]=I[-1,:]=I[:,0]=I[:,-1]=0
    return I.astype(bool)

def lcc(m,n=1):
    lab,k = ndimage.label(m, np.ones((3,3)))
    if k==0: return m
    s = ndimage.sum(m,lab,range(1,k+1))
    return np.isin(lab, np.argsort(s)[::-1][:n]+1)

def walk(sk, seed):
    pts=set(zip(*np.where(sk)[::-1])); path=[seed]; pts.discard(seed); cur=seed; prev=None
    for _ in range(9000):
        cx,cy=cur
        cand=[(cx+dx,cy+dy) for dx in(-1,0,1) for dy in(-1,0,1) if (dx or dy) and (cx+dx,cy+dy) in pts]
        if not cand:
            best,bd=None,1e9
            for dd in range(2,22):
                for dx in range(-dd,dd+1):
                    for dy in range(-dd,dd+1):
                        p=(cx+dx,cy+dy)
                        if p in pts:
                            d=dx*dx+dy*dy
                            if prev: d-=2.5*((cx-prev[0])*dx+(cy-prev[1])*dy)
                            if d<bd: bd,best=d,p
                if best: break
            if best is None: break
            cand=[best]
        if prev and len(cand)>1:
            vx,vy=cx-prev[0],cy-prev[1]
            cand.sort(key=lambda p:-((p[0]-cx)*vx+(p[1]-cy)*vy))
        nx=cand[0]
        for dx in(-1,0,1):
            for dy in(-1,0,1): pts.discard((nx[0]+dx,nx[1]+dy))
        path.append(nx); prev,cur=cur,nx
    return np.array(path,float)

def resample(p,n):
    d=np.r_[0,np.cumsum(np.hypot(*np.diff(p,axis=0).T))]
    s=np.linspace(0,d[-1],n)
    return np.c_[np.interp(s,d,p[:,0]),np.interp(s,d,p[:,1])], d[-1]

def curv(xy,closed):
    n=len(xy); k=np.zeros(n)
    for i in range(n):
        a,b,c=(i-2)%n,i,(i+2)%n
        if not closed and (i<2 or i>n-3): continue
        ax,ay=xy[a];bx,by=xy[b];cx,cy=xy[c]
        ar=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
        dd=np.hypot(bx-ax,by-ay)*np.hypot(cx-bx,cy-by)*np.hypot(ax-cx,ay-cy)
        k[i]=2*ar/dd if dd>1e-6 else 0
    return k

def go(name, src, colorfn, closeiter, seedfn, closed, nnodes):
    im=load(src); r,g,b=im[...,0],im[...,1],im[...,2]
    m=colorfn(r,g,b)
    m=ndimage.binary_dilation(m,iterations=1)
    m=ndimage.binary_closing(m,structure=np.ones((3,3)),iterations=closeiter)
    m=lcc(m,1); m=ndimage.binary_fill_holes(m)
    sk=lcc(thin(m),1)
    ys,xs=np.where(sk); seed=seedfn(xs,ys)
    p=walk(sk,seed)
    print(f'{name}: skel {sk.sum()}  walk {len(p)}  xspan {int(p[:,0].min())}-{int(p[:,0].max())} yspan {int(p[:,1].min())}-{int(p[:,1].max())}')
    # to meters, y-flip
    mm=MPX[name]
    xy=p.copy(); xy[:,0]*=mm; xy[:,1]*=-mm
    xy-=xy.min(axis=0)
    nodes,totm=resample(xy,nnodes)
    k=curv(nodes,closed); ka=np.abs(k)
    minR=1/ka.max() if ka.max()>1e-4 else 9999
    print(f'{name}: LENGTH {totm:.0f} m ({totm/0.3048:.0f} ft)  minR {minR:.1f} m')
    # overlay decimated nodes on map
    ov=im.copy().astype(np.uint8)
    npx=nodes.copy(); npx[:,1]=-(npx[:,1]-nodes[:,1].max()) if False else npx[:,1]
    # convert nodes back to px for overlay
    bx=p[:,0].min()*mm; by=(-p[:,1]).min()*mm
    ndpx=np.c_[(nodes[:,0]+ (p[:,0].min()*mm))/mm, -((nodes[:,1]) + by)/mm]
    for x,y in ndpx.astype(int):
        if 0<=y<ov.shape[0] and 0<=x<ov.shape[1]:
            ov[max(0,y-2):y+3,max(0,x-2):x+3]=[255,0,0]
    Image.fromarray(ov).resize((ov.shape[1]*2,ov.shape[0]*2),Image.NEAREST).save(f'maps/{name}_nodes.png')
    # emit rounded node list
    out=[[round(float(a),1),round(float(c),1)] for a,c in nodes]
    json.dump({'name':name,'length_m':round(totm),'minR_m':round(minR,1),'closed':closed,'nodes':out},
              open(f'{name}_nodes.json','w'))
    print(f'{name}: wrote {name}_nodes.json  ({len(out)} nodes)')
    return nodes,totm

if __name__=='__main__':
    w=sys.argv[1] if len(sys.argv)>1 else 'both'
    if w in ('ax','both'):
        go('ax',AX, lambda r,g,b:(r>150)&(g<115)&(b<115)&(r-g>55),
           2, lambda xs,ys:(int(xs.min()),int(np.median(ys[xs<xs.min()+3]))), False, 85)
    if w in ('en','both'):
        go('en',EN, lambda r,g,b:(r>188)&(g>100)&(g<224)&(b>100)&(b<224)&(r-g>15)&(r-b>9),
           2, lambda xs,ys:tuple(int(v) for v in np.array([xs,ys]).T[np.argmax(xs.astype(float)+(ys>ys.mean())*600)]),
           True, 120)
