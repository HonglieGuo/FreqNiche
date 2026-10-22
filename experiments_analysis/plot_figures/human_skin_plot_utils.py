"""Shared rendering helpers for manuscript spatial and representation figures."""
from pathlib import Path
import hashlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def palette(cats,name):
    cmap=plt.get_cmap(name)
    return {c:cmap(i%cmap.N) for i,c in enumerate(cats)}

def scatter(ax,xy,values,cats,colors,size,alpha=.85):
    for c in cats:
        mask=np.asarray(values)==c
        if mask.any(): ax.scatter(*xy[mask].T,c=[colors[c]],s=size,alpha=alpha,linewidths=0,edgecolors='none',rasterized=True)

def legend(ax,cats,colors,labels=None,ncol=3):
    ax.set_axis_off()
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=4,markerfacecolor=colors[c],markeredgecolor='none',label=s)
             for c,s in zip(cats,labels or [str(c) for c in cats])]
    ax.legend(handles=handles,loc='upper center',ncol=ncol,frameon=False,fontsize=9.5,
              handlelength=.55,handletextpad=.2,columnspacing=.65,borderaxespad=0,labelspacing=.3)

def annotation(ax,text):
    ax.text(.97,.035,text,ha='right',va='bottom',transform=ax.transAxes,fontsize=11.5,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':1.8})
