#%%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

import numpy as np
from sklearn.model_selection import train_test_split

import complexnn

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import multi_gpu_model

act_fn = 'elu'

def CAE(input_shape=None, x=0):

    model = models.Sequential()
    # Block 1
    model.add(complexnn.conv.ComplexConv2D(2**(x+1), (3, 3), activation=act_fn, padding='same', input_shape=input_shape))
    model.add(complexnn.conv.ComplexConv2D(2**(x+1), (3, 3), strides=(2,2), activation=act_fn, padding='same'))
    model.add(complexnn.bn.ComplexBatchNormalization())
    model.add(complexnn.conv.ComplexConv2D(2**(x+2), (3, 3), strides=(2,2), activation=act_fn, padding='same'))
    model.add(complexnn.bn.ComplexBatchNormalization())
    model.add(complexnn.conv.ComplexConv2D(2**(x+3), (3, 3), strides=(2,2), activation=act_fn, padding='same'))
    model.add(complexnn.bn.ComplexBatchNormalization())
    model.add(complexnn.conv.ComplexConv2D(2**(x+4), (3, 3), strides=(2,2), activation=act_fn, padding='same'))
    model.add(complexnn.bn.ComplexBatchNormalization())
        
    model.add(complexnn.conv.ComplexConv2D(2**(x+5), (3, 3), activation=act_fn, padding='same'))
        
    model.add(complexnn.conv.ComplexConv2D(2**(x+4), (3, 3), strides=(2,2), transposed=True, activation=act_fn, padding='same'))
    model.add(complexnn.bn.ComplexBatchNormalization())
    model.add(complexnn.conv.ComplexConv2D(2**(x+3), (3, 3), strides=(2,2), transposed=True, activation=act_fn, padding='same'))
    model.add(complexnn.bn.ComplexBatchNormalization())
    model.add(complexnn.conv.ComplexConv2D(2**(x+2), (3, 3), strides=(2,2), transposed=True, activation=act_fn, padding='same'))
    model.add(complexnn.bn.ComplexBatchNormalization())
    model.add(complexnn.conv.ComplexConv2D(2**(x+1), 3, strides=(2,2), transposed=True, activation = act_fn, padding = 'same'))
    model.add(complexnn.bn.ComplexBatchNormalization())
    model.add(complexnn.conv.ComplexConv2D(2**(x+1), 3, activation = act_fn,  padding = 'same'))
    model.add(complexnn.conv.ComplexConv2D(1 , 3, activation = act_fn,  padding = 'same'))
    return model

def RAE(input_shape=None,x=0):
        
    model = models.Sequential()
    model.add(layers.Conv2D(2**(x+1), (3, 3), activation=act_fn, padding='same', kernel_initializer = 'he_normal', input_shape=input_shape))
    model.add(layers.Conv2D(2**(x+1), (3, 3), activation=act_fn, strides=(2,2), padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2**(x+2), (3, 3), activation=act_fn, strides=(2,2), padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2**(x+3), (3, 3), activation=act_fn, strides=(2,2), padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2**(x+4), (3, 3), activation=act_fn, strides=(2,2), padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2**(x+5), (3, 3), activation=act_fn, padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv2DTranspose(2**(x+4), (3, 3), activation=act_fn, strides=(2,2), padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(2**(x+3), (3, 3), activation=act_fn, strides=(2,2), padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(2**(x+2), (3, 3), activation=act_fn, strides=(2,2), padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(2**(x+1), 3, activation=act_fn, strides=(2,2), padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2**(x+1), 3, activation=act_fn, padding='same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv2D(1, (3, 3), activation=act_fn, padding='same', kernel_initializer = 'he_normal'))
    return model

def par_train(X_train,X_test,model,filename):
    par_model = multi_gpu_model(model, gpus=4)

    par_model.compile(optimizer=optimizers.Adam(), loss='mse', metrics=["mae"])

    csv_cb = keras.callbacks.CSVLogger('../logs/'+filename+'.csv')

    par_model.fit(X_train,
              X_train,
              epochs=100,
              verbose=2,
              batch_size=16,
              shuffle=True,
              validation_data=(X_test, X_test),
              callbacks=[csv_cb])

    par_model.save(filename+'.hd5')

def print_summary(model, filename):
    with open('../descriptions/'+filename + '_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

#%%
print('===================\n===Data Loading====\n===================')
cmplx = np.concatenate([np.load("../patch_data/x_cmplx.npy"), np.load("../patch_data/i_cmplx.npy")])
real = np.concatenate([np.expand_dims(np.load('../patch_data/i_real.npy'),axis=3), np.expand_dims(np.load('../patch_data/x_real.npy'), axis=3)])

print('===================\n==Data Splitting==\n===================')
X_train_cmplx, X_test_cmplx, X_train_real, X_test_real, = train_test_split(cmplx, real, test_size=0.25, random_state=42)
del(cmplx)
del(real)

print('===================\n===Print Summary===\n===================')
#%%
print_summary(CAE((64,64,2),0), 'cmplx_mini')
print_summary(CAE((64,64,2),1), 'cmplx_small')
print_summary(CAE((64,64,2),2), 'cmplx_big')
print_summary(RAE((64,64,1),1), 'real_mini')
print_summary(RAE((64,64,1),2), 'real_small')
print_summary(RAE((64,64,1),3), 'real_big')

#%%
print('===================\n=======Train=======\n===================')
for common_seed in [33,42,12345,914872,552926,175937,528286]:
#for common_seed in [33,]:
    from tensorflow import set_random_seed
    np.random.seed(common_seed)
    set_random_seed(common_seed)

    import complexnn

    import keras
    from keras import models
    from keras import layers
    from keras import optimizers
    from keras.utils import multi_gpu_model

    #print('===================\n=====Train R 0====\n===================')
    #par_train(X_train_real, X_test_real,RAE((None,None,1),1),'real_mini') # Real Mini
    print('===================\n=====Train R S====\n===================')
    par_train(X_train_real, X_test_real,RAE((None,None,1),2),'real_small') # Real Small
    print('===================\n=====Train C S====\n===================')
    par_train(X_train_cmplx, X_test_cmplx,CAE((None,None,2),1),'cmplx_small') # Complex Small
    print('===================\n=====Train C L====\n===================')
    par_train(X_train_cmplx, X_test_cmplx,CAE((None,None,2),2),'cmplx_big') # Complex Large
    print('===================\n=====Train R L====\n===================')
    par_train(X_train_real, X_test_real,RAE((None,None,1),3),'real_large') # Real Large
    #print('===================\n=====Train C 0====\n===================')
    #par_train(X_train_cmplx, X_test_cmplx,CAE((None,None,2),0),'cmplx_mini') # Complex Mini

