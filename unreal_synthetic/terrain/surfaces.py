"""Ramp profile shared by terrain meshing and validation."""
import numpy as np


def ramp_blend(t,shoulder=.16):
    """C1 end shoulders with a constant-grade middle, rather than a sharp kink."""
    t=np.clip(t,0,1);a=shoulder
    low=t/2-a/(2*np.pi)*np.sin(np.pi*t/a)
    q=1-t;high=1-a-(q/2-a/(2*np.pi)*np.sin(np.pi*q/a))
    return np.where(t<a,low,np.where(t>1-a,high,t-a/2))/(1-a)
