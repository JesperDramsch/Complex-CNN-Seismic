#%%
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%%
# Load data
keys = ['truth', 'small_complex', 'small_real', 'big_complex', 'big_real']

with np.load("predictions.npz") as data:
    x,y = data['truth'].shape
    seismic = np.zeros((x,y,5))
    for i, q in enumerate(keys):
        seismic[:,:,i] = data[q]

#%%
# Prepare Patches
f3_offset = 0.832 # Top crop of data

bottom_xt = [350,675,130,250]
top_xt = [50,300,30,100]
silent_xt = [500, 695,70, 120]

bottom = seismic[bottom_xt[2]:bottom_xt[3],bottom_xt[0]:bottom_xt[1],:]
top = seismic[top_xt[2]:top_xt[3],top_xt[0]:top_xt[1],:]
silent = seismic[silent_xt[2]:silent_xt[3],silent_xt[0]:silent_xt[1],:]


#%% 
# Generate annotated ground truth
fig, ax = plt.subplots(figsize=(20,7))
im = ax.imshow(seismic[:,:,0], vmin=-1, vmax=1, aspect='auto', extent=[0,seismic.shape[1]*25,seismic.shape[0]*.004+f3_offset,f3_offset])
ax.set_title("Full Seismic Data")
ax.set_ylabel("Time [s]")
ax.set_xlabel("Offset [m]")

bc = 'k'
tc = 'w'
sc = 'r'

brect = patches.Rectangle((bottom_xt[0]*25,bottom_xt[2]*0.004+f3_offset),(bottom_xt[1]-bottom_xt[0])*25,(bottom_xt[3]-bottom_xt[2])*0.004,linewidth=1.25, edgecolor=bc,facecolor='none')
trect = patches.Rectangle((top_xt[0]*25,top_xt[2]*0.004+f3_offset),(top_xt[1]-top_xt[0])*25,(top_xt[3]-top_xt[2])*0.004,linewidth=1, edgecolor=tc,facecolor='none')
srect = patches.Rectangle((silent_xt[0]*25,silent_xt[2]*0.004+f3_offset),(silent_xt[1]-silent_xt[0])*25,(silent_xt[3]-silent_xt[2])*0.004,linewidth=1, edgecolor=sc,facecolor='none')

# Add the patch to the Axes
ax.add_patch(brect)
ax.add_patch(trect)
ax.add_patch(srect)

ax.annotate('Top', ((top_xt[0]+10)*25,f3_offset+(top_xt[2]+10)*0.004), color=tc, weight='bold', fontsize=16, ha='left', va='center')
ax.annotate('Bottom', ((bottom_xt[0]+10)*25,f3_offset+(bottom_xt[2]+10)*0.004), color=bc, weight='bold', fontsize=16, ha='left', va='center')
ax.annotate('Silent', ((silent_xt[0]+10)*25,f3_offset+(silent_xt[2]+10)*0.004), color=sc, weight='bold', fontsize=16, ha='left', va='center')

fig.colorbar(im, ax=ax)
fig.tight_layout()
# fig.show()
fig.savefig(f"figures/seismic.png", bbox_inches='tight', dpi=200)


#%%
plt.imshow(seismic[:,:,0])


#%%
plt.imshow(seismic[50:150,500:,0])

#%%
def rms(a,b):
    return np.sqrt(np.power(a-b,2).mean())

def mae(a,b):
    return np.abs(a-b).mean()

def format_seis(data, keys, keyword=""):
    for i in range(data.shape[-1]):
        print(f"{keys[i]:13s} {keyword}:\t | RMS: {rms(data[:,:,0],data[:,:,i]):.4f}\t | MAE: {mae(data[:,:,0],data[:,:,i]):.4f}")

def plot_seis(data, keys, size=(20,7), keyword="", xtent=None):
    for i in range(data.shape[-1]):
        fig, ax = plt.subplots(figsize=size)
        if xtent:
            im = ax.imshow(data[:,:,i], vmin=-1, vmax=1, aspect='auto', extent=[xtent[0]*25,xtent[1]*25,xtent[3]*.004+f3_offset,xtent[2]*.004+f3_offset])
        else:
            im = ax.imshow(data[:,:,i], vmin=-1, vmax=1, aspect='auto', extent=[0,data.shape[1]*25,data.shape[0]*.004+f3_offset,f3_offset])
        ax.set_title(keys[i].replace("_"," ").title())
        ax.set_ylabel("Time [s]")
        ax.set_xlabel("Offset [m]")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        # fig.show()
        fig.savefig(f"figures/{keyword}_{keys[i]}.png", bbox_inches='tight', dpi=200)

def plot_fk(data, keys, size=(5,5), keyword=""):
    for i in range(data.shape[-1]):
        M, N = data[:,:,i].shape
        fft2 = fftpack.fft2(data[:,:,i])
        f_mag = np.abs(fft2)
        f_mag = fftpack.fftshift(f_mag)
        f_mag = np.log(1 + f_mag)
        fig, ax = plt.subplots(figsize=size)
        q, p = fftpack.fftfreq(M, d=.004), fftpack.fftfreq(N, d=25/1000)
        im = ax.imshow(f_mag[:M//2,N//2:9*N//16], aspect='auto', vmin=0, vmax=8,
        extent=(0, max(p)/8, 0, max(q)))
        ax.set_title(keys[i].replace("_"," ").title())
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Wavenumber [km$^{-1}$]")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        # fig.show()
        fig.savefig(f"figures/{keyword}_fk_{keys[i]}.png", bbox_inches='tight', dpi=200)

def plot_fk_diff(data, keys, size=(5,5), keyword=""):
    for i in range(1,data.shape[-1]):
        M, N = data[:,:,i].shape
        fft2 = fftpack.fft2(data[:,:,i])
        f_mag = np.abs(fft2) - np.abs(fftpack.fft2(data[:,:,0]))
        f_mag = fftpack.fftshift(f_mag)
        #f_mag = np.log(1 + f_mag)
        vm = np.abs(f_mag).max()
        q, p = fftpack.fftfreq(M, d=.004), fftpack.fftfreq(N, d=25/1000)
        fig, ax = plt.subplots(figsize=size)
        im = ax.imshow(f_mag[:M//2,:], aspect='auto', cmap='RdBu', vmin=-vm, vmax=vm,
        extent=(min(p), max(p), 0, max(q)))
        ax.set_title(keys[i].replace("_"," ").title())
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Wavenumber [km$^{-1}$]")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        # fig.show()
        fig.savefig(f"figures/{keyword}_fk_diff_{keys[i]}.png", bbox_inches='tight', dpi=200)

print(seismic.shape[1], [x*seismic.shape[1]//y for x,y in [(1,2), (3,4), (5,8), (9,16)]], [x*seismic.shape[1]//y-seismic.shape[1]//2 for x,y in [(1,2), (3,4), (5,8), (9,16)]])

#%%
# Let's first calculate the rms and mae on the full seismic and the cutouts.
format_seis(seismic, keys, "full")

#%%
format_seis(bottom, keys, "bottom")

#%%
format_seis(top, keys, "top")

#%%
format_seis(silent, keys, "silent")


#%%
# Plot seismic
plot_seis(seismic, keys, (20,7), "full")

#%%
plot_seis(bottom, keys,(10,6), "bottom", bottom_xt)

#%%
plot_seis(top,keys,(10,6), "top", top_xt)

#%%
plot_seis(silent,keys,(10,6), "silent", silent_xt)


#%%
# Plot FK representation of Seismic
plot_fk(seismic,keys,keyword="full")

#%%
plot_fk(top,keys,keyword="top")

#%%
plot_fk(bottom,keys,keyword="bottom")

#%%
plot_fk(silent,keys,keyword="silent")


#%%
# Plot difference of FK images
plot_fk_diff(seismic,keys,keyword="full") 

#%%
plot_fk_diff(top,keys,keyword="top") 

#%%
plot_fk_diff(bottom,keys,keyword="bottom")

#%%
plot_fk_diff(silent,keys,keyword="silent")
