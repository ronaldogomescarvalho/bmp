"""
rPPG HRV Monitor — Servidor Web
================================
Métodos implementados baseados nos papers originais:
  GREEN : Verkruysse et al. (2008) Optics Express
  CHROM : De Haan & Jeanne (2013) IEEE TBME
  POS   : Wang et al. (2017) IEEE TBME
  ICA   : Poh et al. (2010) Optics Express
  PCA   : Lewandowska et al. (2011) FedCSIS
  LGI   : Pilz et al. (2018) CVPRW
  PBV   : De Haan & Van Leest (2014) Physiol. Meas.
  SSR   : Wang et al. (2015) IEEE TBME
  OMIT  : Alvarez Casado & Bordallo Lopez (2022) arXiv
"""
import os, base64, time
import numpy as np
from io import BytesIO
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

FILTER_LOW_HZ=0.7; FILTER_HIGH_HZ=3.5; FILTER_ORDER=3
PEAK_MIN_DISTANCE_SEC=0.3; PEAK_MAX_DISTANCE_SEC=1.5
MIN_SAMPLES_FOR_BPM=60; MIN_SAMPLES_FOR_HRV=150
FOREHEAD_CENTRAL=[10,67,109,108,151,337,338,297]

app=Flask(__name__)
app.config["SECRET_KEY"]=os.environ.get("SECRET_KEY","rppg-secret-2024")
socketio=SocketIO(app,cors_allowed_origins="*",async_mode="eventlet")

MODEL_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"face_landmarker.task")
base_options=mp_tasks_python.BaseOptions(model_asset_path=MODEL_PATH)
options=mp_tasks_vision.FaceLandmarkerOptions(base_options=base_options,num_faces=1,
    min_face_detection_confidence=0.5,min_face_presence_confidence=0.5,min_tracking_confidence=0.5)
face_landmarker=mp_tasks_vision.FaceLandmarker.create_from_options(options)
sessions={}

def new_session(method="CHROM"):
    return {"rgb_signal":[],"timestamps":[],"method":method}

def bandpass(sig,fs):
    nyq=fs/2.0; lo=max(FILTER_LOW_HZ/nyq,0.001); hi=min(FILTER_HIGH_HZ/nyq,0.999)
    if lo>=hi or len(sig)<15: return sig
    b,a=butter(FILTER_ORDER,[lo,hi],btype="band")
    pl=min(len(sig)-1,3*max(len(b),len(a)))
    return filtfilt(b,a,sig,padlen=pl)

def detrend_norm(s):
    x=np.arange(len(s)); s=s-np.polyval(np.polyfit(x,s,1),x)
    sd=np.std(s); return (s-np.mean(s))/sd if sd>1e-9 else s

def best_snr(comps,fs):
    best,bsig=-1,comps[0]
    for c in comps:
        cf=bandpass(c,fs); n=len(cf)
        yf=np.abs(fft(cf*np.hanning(n)))[:n//2]; xf=fftfreq(n,1.0/fs)[:n//2]
        ib=(xf>=FILTER_LOW_HZ)&(xf<=FILTER_HIGH_HZ); ob=~ib&(xf>0)
        if not np.any(ib) or not np.any(ob): continue
        snr=np.sum(yf[ib]**2)/(np.sum(yf[ob]**2)+1e-9)
        if snr>best: best,bsig=snr,cf
    return bsig

def estimate_bpm(sig,fs):
    n=len(sig)
    if n<2: return None
    yf=np.abs(fft(sig*np.hanning(n)))[:n//2]; xf=fftfreq(n,1.0/fs)[:n//2]
    v=(xf>=0.67)&(xf<=3.0)
    if not np.any(v): return None
    yv=yf[v]; xv=xf[v]
    freq=xv[np.argmax(yv)]; bpm=freq*60.0
    if bpm>100:
        hf=freq/2.0; hv=(xf>=hf*0.9)&(xf<=hf*1.1)
        if np.any(hv) and np.max(yf[hv])>=np.max(yv)*0.4:
            bpm=bpm/2.0
    return bpm

def compute_hrv(sig,fs):
    if len(sig)<fs*5: return None,None
    peaks,_=find_peaks(sig,distance=max(int(PEAK_MIN_DISTANCE_SEC*fs),1),
        height=np.median(sig),prominence=np.std(sig)*0.3)
    if len(peaks)<3: return None,None
    rr=np.diff(peaks)/fs*1000.0
    rr=rr[(rr>=PEAK_MIN_DISTANCE_SEC*1000)&(rr<=PEAK_MAX_DISTANCE_SEC*1000)]
    if len(rr)<2: return None,None
    return rr.tolist(),float(np.sqrt(np.mean(np.diff(rr)**2)))

def readiness_score(bpm,rmssd):
    if bpm is None or rmssd is None: return None
    bs=50.0 if 55<=bpm<=75 else (max(0,50-(55-bpm)*2.5) if bpm<55 else max(0,50-(bpm-75)*2.0))
    hs=50.0 if rmssd>=80 else ((rmssd-20)/60.0*50.0 if rmssd>=20 else max(0,rmssd/20.0*10.0))
    return round(min(100,max(0,bs+hs)),1)

# ── Métodos rPPG ──────────────────────────────────────────────────────────────

def method_GREEN(rgb,fs):
    """Verkruysse et al. (2008) Optics Express 16(26):21434."""
    return bandpass(detrend_norm(rgb[:,1]),fs)

def method_CHROM(rgb,fs):
    """De Haan & Jeanne (2013) IEEE TBME 60(10):2878."""
    m=np.mean(rgb,axis=0); m=np.where(m==0,1e-6,m)
    rn,gn,bn=rgb[:,0]/m[0],rgb[:,1]/m[1],rgb[:,2]/m[2]
    Xs=3*rn-2*gn; Ys=1.5*rn+gn-1.5*bn
    Xf=bandpass(Xs,fs); Yf=bandpass(Ys,fs)
    a=np.std(Xf)/(np.std(Yf) if np.std(Yf)>1e-6 else 1.0)
    return Xf-a*Yf

def method_POS(rgb,fs):
    """Wang et al. (2017) IEEE TBME 64(7):1479."""
    m=np.mean(rgb,axis=0); m=np.where(m==0,1e-6,m)
    rn,gn,bn=rgb[:,0]/m[0],rgb[:,1]/m[1],rgb[:,2]/m[2]
    S1=rn-gn; S2=rn+gn-2*bn
    S1f=bandpass(S1,fs); S2f=bandpass(S2,fs)
    b=np.std(S1f)/(np.std(S2f) if np.std(S2f)>1e-6 else 1.0)
    return detrend_norm(S1f+b*S2f)

def method_ICA(rgb,fs):
    """Poh et al. (2010) Optics Express 18(10):10762."""
    C=rgb.T.astype(float); m=np.mean(C,axis=1,keepdims=True); s=np.std(C,axis=1,keepdims=True)
    s=np.where(s==0,1e-6,s); Cn=(C-m)/s
    U,S,_=np.linalg.svd(Cn,full_matrices=False); S=np.where(S<1e-6,1e-6,S)
    Z=(np.diag(1.0/S)@U.T)@Cn
    return best_snr(Z,fs)

def method_PCA(rgb,fs):
    """Lewandowska et al. (2011) FedCSIS pp.405."""
    C=rgb.T.astype(float); Cn=C-np.mean(C,axis=1,keepdims=True)
    _,vecs=np.linalg.eigh(np.cov(Cn)); comps=vecs[:,::-1].T@Cn
    return best_snr(comps,fs)

def method_LGI(rgb,fs):
    """Pilz et al. (2018) CVPRW pp.1254."""
    m=np.mean(rgb,axis=0); m=np.where(m==0,1e-6,m)
    rn,gn,bn=rgb[:,0]/m[0],rgb[:,1]/m[1],rgb[:,2]/m[2]
    U=np.stack([rn,gn,bn],axis=0); n=np.linalg.norm(U,axis=0,keepdims=True)
    n=np.where(n==0,1e-6,n); Un=U/n; w=np.std(Un,axis=1)
    return bandpass(detrend_norm(w[0]*Un[0]-w[1]*Un[1]),fs)

def method_PBV(rgb,fs):
    """De Haan & Van Leest (2014) Physiol. Meas. 35(9):1913."""
    pbv=np.array([0.3,0.45,0.25]); pbv/=np.linalg.norm(pbv)
    C=rgb.T.astype(float); m=np.mean(C,axis=1,keepdims=True); m=np.where(m==0,1e-6,m); Cn=C/m
    bvp=pbv@Cn; mot=np.linalg.norm(Cn-np.outer(pbv,pbv@Cn),axis=0)
    mot=np.where(mot==0,1e-6,mot)
    return bandpass(detrend_norm(bvp/(1.0+mot)),fs)

def method_SSR(rgb,fs):
    """Wang et al. (2015) IEEE TBME 63(9):1974."""
    N=len(rgb)
    if N<4: return np.zeros(N)
    bvp=np.zeros(N); win=min(32,N//2); dc=np.array([1,1,1])/np.sqrt(3)
    for i in range(win,N):
        seg=rgb[i-win:i].T.astype(float); m=np.mean(seg,axis=1,keepdims=True)
        m=np.where(m==0,1e-6,m)
        try:
            U,_,_=np.linalg.svd(seg/m,full_matrices=False)
            proj=U[:,0]-np.dot(U[:,0],dc)*dc; bvp[i]=np.dot(proj,(seg/m)[:,-1])
        except Exception: pass
    return bandpass(detrend_norm(bvp),fs)

def method_OMIT(rgb,fs):
    """Alvarez Casado & Bordallo Lopez (2022) arXiv:2202.04101."""
    C=rgb.T.astype(float); m=np.mean(C,axis=1,keepdims=True); m=np.where(m==0,1e-6,m)
    Cn=C/m-1.0; U,_,_=np.linalg.svd(Cn,full_matrices=False)
    md=U[:,0:1]; clean=Cn-md@(md.T@Cn)
    return bandpass(detrend_norm(np.array([0.7,1.0,0.3])@clean),fs)

METHODS={
    "GREEN":method_GREEN,"CHROM":method_CHROM,"POS":method_POS,
    "ICA":method_ICA,"PCA":method_PCA,"LGI":method_LGI,
    "PBV":method_PBV,"SSR":method_SSR,"OMIT":method_OMIT,
}

# ── Processamento de frame ────────────────────────────────────────────────────
def process_frame(image_data_b64,session):
    try:
        _,encoded=image_data_b64.split(",",1)
        frame=np.array(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))
    except Exception as e: return {"error":str(e)}

    h,w,_=frame.shape
    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=frame)
    res=face_landmarker.detect(mp_img)
    if not res.face_landmarks:
        return {"face_detected":False,"samples":len(session["rgb_signal"])}

    lm=res.face_landmarks[0]
    pts=np.array([[int(lm[i].x*w),int(lm[i].y*h)] for i in FOREHEAD_CENTRAL],dtype=np.int32)
    x1,y1=max(0,pts[:,0].min()),max(0,pts[:,1].min())
    x2,y2=min(w,pts[:,0].max()),min(h,pts[:,1].max())
    roi=frame[y1:y2,x1:x2]
    if roi.size==0:
        return {"face_detected":True,"samples":len(session["rgb_signal"])}

    session["rgb_signal"].append([float(np.mean(roi[:,:,c])) for c in range(3)])
    session["timestamps"].append(time.time())

    n=len(session["rgb_signal"]); ts=session["timestamps"]
    fs=(n-1)/(ts[-1]-ts[0]) if n>=2 and ts[-1]>ts[0] else 30.0
    result={"face_detected":True,"samples":n,"fps":round(fs,1)}

    fn=METHODS.get(session.get("method","CHROM"),method_CHROM)
    rgb=np.array(session["rgb_signal"])

    if n>=MIN_SAMPLES_FOR_BPM:
        try:
            bpm=estimate_bpm(fn(rgb,fs),fs)
            if bpm and 40<=bpm<=180: result["bpm"]=round(bpm,1)
        except Exception: pass

    if n>=MIN_SAMPLES_FOR_HRV:
        try:
            rr,rmssd=compute_hrv(fn(rgb,fs),fs)
            if rmssd is not None:
                result["rmssd"]=round(rmssd,1)
                result["score"]=readiness_score(result.get("bpm"),rmssd)
        except Exception: pass

    return result

# ── Rotas ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@socketio.on("connect")
def on_connect():
    sessions[_sid()]=new_session(); emit("connected",{"msg":"ok"})

@socketio.on("disconnect")
def on_disconnect(): sessions.pop(_sid(),None)

@socketio.on("reset")
def on_reset(data=None):
    method=data.get("method","CHROM") if isinstance(data,dict) else "CHROM"
    sessions[_sid()]=new_session(method=method); emit("reset_ok")

@socketio.on("frame")
def on_frame(data):
    sid=_sid()
    if sid not in sessions: sessions[sid]=new_session()
    emit("result",process_frame(data["image"],sessions[sid]))

def _sid():
    from flask import request
    return request.sid

if __name__=="__main__":
    socketio.run(app,host="0.0.0.0",port=8080,debug=False,allow_unsafe_werkzeug=True)
