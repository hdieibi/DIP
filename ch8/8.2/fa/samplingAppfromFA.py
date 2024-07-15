# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 21:15:18 2018
sampling app and geo from eigen-value and eigen vector of PCA
@author: andy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import color
import scipy.io as sio 
from sklearn.decomposition import PCA, FactorAnalysis
from mywarper import warp
from utils import plot
import imageio


trainmatpath='./img_lm_train.npz'
traindata = np.load(trainmatpath)

Vtrain=traindata['Vtrain']
Ntr,D=Vtrain.shape
LMtrain=traindata['LMtrain']
HStrain=traindata['HStrain']
HW=int(np.sqrt(D))
Ka=50
Kg=10
numlds=68
nsamp=20
Plotslmgeo=1;Plotsgeo=1;Plotsapp=1
"""
%##############samping from geometric landmarks########################
"""
vec_LMtrain=np.reshape(LMtrain,[Ntr,numlds*2])
pca_g=PCA(n_components=0.999)
pca_g.fit(vec_LMtrain)
eigvalueGeo=pca_g.explained_variance_
eigvectorGeo=pca_g.components_.T
vec_LM_mean_tr = pca_g.mean_
coff=np.matmul((vec_LMtrain-np.ones((Ntr,1))*vec_LM_mean_tr),eigvectorGeo[:,0:Kg])
eigengaxisunit=4.0*np.std(coff,axis=0)
eigengaxisunit1=2.5*np.std(coff,axis=0)
eigengaxismean=np.mean(coff,0)
sampleigngvalue=np.zeros((nsamp*Kg,Kg))
sampling_G=np.zeros((nsamp*Kg,numlds*2))
for i in range(Kg):
    if i==0:
        sampleigngvalue[i*nsamp:i*nsamp+nsamp,i]=np.linspace(-eigengaxisunit1[i],eigengaxisunit1[i],nsamp)
    else:
        sampleigngvalue[i*nsamp:i*nsamp+nsamp,i]=np.linspace(-eigengaxisunit[i],eigengaxisunit[i],nsamp)
sampling_G=np.matmul(sampleigngvalue,eigvectorGeo[:,0:Kg].T)+1*np.ones((nsamp*Kg,1))*vec_LM_mean_tr
sampling_LM=np.reshape(sampling_G,[nsamp*Kg,numlds,2])
if Plotslmgeo:
    for i in range(int(np.ceil(Kg/10))):
        plt.figure(figsize=(nsamp, Kg))
        plt.clf()
        gs = gridspec.GridSpec(Kg, nsamp)
        gs.update(wspace=0.05, hspace=0.05)
        for ii, sample in enumerate(sampling_LM[i*10*nsamp:(i+1)*10*nsamp,:,:]):
            ax = plt.subplot(gs[ii])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.scatter(sample[:,0],HW-sample[:,1],s=10,alpha=0.8,c='red')
            plt.show
'''
%%%%###################sampling from appearance#####################
%For train image, first align the images by warping their landmarks into the mean position
'''
VtrainAlign = np.zeros((Ntr,D))
HStrainAlign=np.zeros((Ntr,HW,HW,2))
V_mean_tr = np.mean(Vtrain,0)
LM_mean_tr=np.reshape(np.mean(LMtrain,0),[numlds,2])
for i in range(Ntr):
#for i in range(15):
    Image=np.reshape(Vtrain[i,:],(HW,HW,1))
    #Image=np.transpose(Image,(1,0,2))
    H=np.reshape(HStrain[i,:,:,0],(HW,HW,1))
    S=np.reshape(HStrain[i,:,:,1],(HW,HW,1))
    originalMarks=np.reshape(LMtrain[i,:,:],[numlds,2])
    imgWarped=warp(Image,originalMarks,LM_mean_tr)
    HWarped=warp(H,originalMarks,LM_mean_tr)
    SWarped=warp(S,originalMarks,LM_mean_tr)
    VtrainAlign[i,:]=np.reshape(imgWarped,[1,D])
    HStrainAlign[i,:,:,:]=np.concatenate((np.reshape(HWarped,[HW,HW,1]),np.reshape(SWarped,[HW,HW,1])),axis=2)
'''
%%%%%%##########warping the aligned faces to the generated landmarks######
'''
# idp=9
# Image=np.reshape(VtrainAlign[idp,:],(HW,HW,1))
# H=np.reshape(HStrainAlign[idp,:,:,0],(HW,HW,1))
# S=np.reshape(HStrainAlign[idp,:,:,1],(HW,HW,1))
# sampling=np.zeros((nsamp*Kg,HW,HW,3))
# samplingseq=[]
# for i in range(nsamp*Kg):
#     imgWarped=warp(Image,LM_mean_tr,np.reshape(sampling_LM[i,:,:],(numlds,2)))
#     HWarped=warp(H,LM_mean_tr,np.reshape(sampling_LM[i,:,:],(numlds,2)))
#     SWarped=warp(S,LM_mean_tr,np.reshape(sampling_LM[i,:,:],(numlds,2)))
#     sampling[i,:,:,:]=np.reshape(color.hsv2rgb(np.concatenate((HWarped,SWarped,imgWarped),axis=2)),(1,HW,HW,3))
#     samplingseq.append(np.array(sampling[i,:,:,:] * 255., dtype=np.uint8))
# if Plotsgeo:
#     for i in range(int(np.ceil(Kg/10))):
#         samplepage=sampling[i*10*nsamp:(i+1)*10*nsamp,:,:,:]
#         fig = plot(samplepage,10,np.int32(nsamp),3,HW, HW)
#         plt.show
#     output_file = 'samplinggeoid{}.gif'.format(str(idp).zfill(3))
#     imageio.mimsave(output_file,samplingseq , duration=0.1)
'''
and then compute the eigen-faces (appearance) from these aligned images
'''
# pca_a=PCA(n_components=0.999)
fa = FactorAnalysis(100)
fa.fit(VtrainAlign)
# eigvalueApp=pca_a.explained_variance_
eigvectorApp=fa.components_.T
VtrainAlign_mean = fa.mean_
HStrainAlign_mean = np.mean(HStrainAlign,0)
coff=np.matmul((VtrainAlign-np.ones((Ntr,1))*VtrainAlign_mean),eigvectorApp[:,0:Ka])
eigenaxisunit=5*np.std(coff,0)
eigenaxismean=np.mean(coff,0)
sampleigngvalue=np.zeros((nsamp*Ka,Ka))
sampling_V=np.zeros((nsamp*Ka,D))
for i in range(Ka):
    sampleigngvalue[i*nsamp:i*nsamp+nsamp,i]=np.linspace(-eigenaxisunit[i],eigenaxisunit[i],nsamp)    
sampling_V=np.matmul(sampleigngvalue,eigvectorApp[:,0:Ka].T)+1*np.ones((nsamp*Ka,1))*VtrainAlign_mean
HSAlign_mean=np.tile(HStrainAlign_mean,(nsamp*Ka,1,1,1))
sampling_hsv=np.concatenate((HSAlign_mean,np.reshape(sampling_V,[nsamp*Ka,HW,HW,1])),3)
sampling_rgb=np.zeros((nsamp*Ka,HW,HW,3))
for i in range(nsamp*Ka):
    sampling_rgb[i,:,:,:]=color.hsv2rgb(sampling_hsv[i,:,:,:])
if Plotsapp:
    for i in range(int(np.ceil(Ka/10))):
        samplepage=sampling_rgb[i*10*nsamp:(i+1)*10*nsamp,:,:,:]
        fig = plot(samplepage,10,np.int32(nsamp),3,HW, HW)
        plt.show
    samplingseq=[]
    for i in range(nsamp*Ka):
        # samplingseq.append(np.array(sampling_rgb[i,:,:,:] * 255., dtype=np.uint8))
        imageio.imsave("sample_app_fa/{}.png".format(i), np.array(sampling_rgb[i,:,:,:] * 255., dtype=np.uint8))
    # output_file = 'samplingapp_fa.gif'
    # imageio.mimsave(output_file,samplingseq , duration=0.1)

