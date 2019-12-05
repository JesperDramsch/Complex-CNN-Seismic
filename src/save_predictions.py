# coding: utf-8

import keras
import numpy as np
import complexnn

from sklearn.model_selection import train_test_split

CCAE = keras.models.load_model(
    "cmplx_big33.hd5",
    custom_objects={
        "ComplexConv2D": complexnn.conv.ComplexConv2D,
        "ComplexBatchNormalization": complexnn.bn.ComplexBatchNormalization,
    },
)

sCCAE = keras.models.load_model(
    "cmplx_small33.hd5",
    custom_objects={
        "ComplexConv2D": complexnn.conv.ComplexConv2D,
        "ComplexBatchNormalization": complexnn.bn.ComplexBatchNormalization,
    },
)


RCAE = keras.models.load_model("real_small33.hd5")
bRCAE = keras.models.load_model("real_big33.hd5")


big_seismic = np.rot90(np.load("../data/test_once/test1_seismic.npy")[0:1,:,:], axes=(1,2), k=3)

from scipy.signal import hilbert
tmp_complex = hilbert(np.squeeze(big_seismic), axis=0)
big_complex = np.expand_dims(np.stack([np.real(tmp_complex), np.imag(tmp_complex)],axis=2),axis=0)


bc_pred = np.squeeze(CCAE.predict(big_complex))[:255,:701]
sc_pred = np.squeeze(sCCAE.predict(big_complex))[:255,:701]

sr_pred = np.squeeze(RCAE.predict(np.expand_dims(big_seismic,axis=3)))[:255,:701]
br_pred = np.squeeze(bRCAE.predict(np.expand_dims(big_seismic,axis=3)))[:255,:701]

np.savez('predictions.npz', truth=big_seismic[0], small_complex=sc_pred[:, :, 0], big_complex=bc_pred[:, :, 0], small_real=np.squeeze(sr_pred), big_real=np.squeeze(br_pred))
