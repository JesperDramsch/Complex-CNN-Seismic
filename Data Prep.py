
# coding: utf-8

# In[1]:


import numpy as np
from skimage.util.shape import view_as_windows
from skimage.io import imsave
from scipy.signal import hilbert


# In[2]:


train_seismic = np.load("data/train/train_seismic.npy")
#train_labels = np.load("data/train/train_labels.npy")


# In[3]:


train_hilbert = np.zeros_like(train_seismic, dtype=np.complex)
for x in range(train_hilbert.shape[0]):
    for y in range(train_hilbert.shape[1]):
        train_hilbert[x,y,:] = hilbert(train_seismic[x,y,:])


# In[4]:


#np.min(np.abs(train_hilbert))


# In[5]:


train_complex = train_hilbert
train_complex = train_hilbert-train_seismic


# In[10]:



patch_size = 64 
patch_size = 64

stride = 8


# In[11]:


patch_shape = (1, patch_size, patch_size)


# In[12]:


real_data = view_as_windows(train_seismic, patch_shape, step=stride)
cmplx_data = view_as_windows(train_complex, patch_shape, step=stride)
#label_data = view_as_windows(train_labels, patch_shape, step=stride)


# In[13]:


real_data.shape


# In[14]:


p = .9
val_split = np.random.choice(a=[False, True], size=real_data.shape[0:3], p=[p, 1-p])


# In[15]:


real = []
cmplx = []
#label = []
#label_patch = []

for a in range(real_data.shape[0]):
    for b in range(real_data.shape[1]):
        for c in range(real_data.shape[2]):
            real.append(np.squeeze(real_data[a,b,c,0,:,:]).T)
            cmplx_patch = np.squeeze(cmplx_data[a,b,c,0,:,:]).T
            cmplx.append(np.stack([np.real(cmplx_patch), np.imag(cmplx_patch)], axis=2))
            #label.append(np.squeeze(label_data[a,b,c,patch_shape[0]//2,patch_shape[1]//2,patch_shape[2]//2]).T)    
            #label_patch.append(np.squeeze(label_data[a,b,c,0,:,:]).T)

            """
            if val_split[a,b,c]:
                where = "val"
            else:
                where = "train"
            label = label_data[a,b,c,patch_shape[0]//2,patch_shape[1]//2,patch_shape[2]//2]
            imsave("image_data/"+where+"/"+str(label)+"/"+str(0*1000+a*100+b*10+c)+".png",np.squeeze(patch_data[a,b,c,0,:,:]).T)
            imsave("image_data/target/"+str(label)+"/"+str(0*1000+a*100+b*10+c)+".png",np.squeeze(label_data[a,b,c,0,:,:]).T)
            """


# In[16]:


np.save('patch_data/i_real.npy', real)


# In[17]:


np.save('patch_data/i_cmplx.npy', cmplx)


# In[18]:


#np.save('patch_data/i_label.npy', label)


#np.save('patch_data/i_label_patches.npy', label_patch)

# In[19]:


patch_shape = (patch_size, 1, patch_size)


# In[20]:


real_data = view_as_windows(train_seismic, patch_shape, step=stride)
cmplx_data = view_as_windows(train_complex, patch_shape, step=stride)
#label_data = view_as_windows(train_labels, patch_shape, step=stride)


# In[21]:


real = []
cmplx = []
#label = []
#label_patch = []

for a in range(real_data.shape[0]):
    for b in range(real_data.shape[1]):
        for c in range(real_data.shape[2]):
            real.append(np.squeeze(real_data[a,b,c,:,0,:]).T)
            cmplx_patch = np.squeeze(cmplx_data[a,b,c,:,0,:]).T
            cmplx.append(np.stack([np.real(cmplx_patch), np.imag(cmplx_patch)], axis=2))
            #label.append(np.squeeze(label_data[a,b,c,patch_shape[0]//2,patch_shape[1]//2,patch_shape[2]//2]).T)
            
            #label_patch.append(np.squeeze(label_data[a,b,c,:,0,:]).T)

            """
            if val_split[a,b,c]:
                where = "val"
            else:
                where = "train"
            label = label_data[a,b,c,patch_shape[0]//2,patch_shape[1]//2,patch_shape[2]//2]
            imsave("image_data/"+where+"/"+str(label)+"/"+str(0*1000+a*100+b*10+c)+".png",np.squeeze(patch_data[a,b,c,:,0,:]).T)
            imsave("image_data/target/"+str(label)+"/"+str(0*1000+a*100+b*10+c)+".png",np.squeeze(label_data[a,b,c,:,0,:]).T)
            """


# In[22]:


np.save('patch_data/x_real.npy', real)


# In[23]:


np.save('patch_data/x_cmplx.npy', cmplx)


# In[24]:


#np.save('patch_data/x_label.npy', label)

#np.save('patch_data/x_label_patches.npy', label_patch)

