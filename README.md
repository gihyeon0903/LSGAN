# LSGAN
<br/>

### Datasets
-------------------
<a href='https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'>CelebA</a> : 약 20k개의 연예인 얼굴 datasets

<p align="center">
  <img src="./result/celeba.jfif" width="300" height="200"/>
</p>

### Model
-------------------
LSGAN

<a href='https://drive.google.com/drive/folders/1danZgK_eBRnnBUR43JkTtwnAptNxPJg4?usp=drive_link'>Weights</a>
<p align="center">
  <img src="./result/model.png" width="700" height="150"/>
</p>

DCGAN 모델이서 Tanh를 제거

### Train
-------------------
epochs : 10, learning rate : 0.0005, optimizer : Adam(Beta1, 2 = 0.5), Loss : Least Square Loss <br>
Man Data

<br/>

### Result
-------------------
#### 1. D, G Loss
<p align="center">
  <img src="./result/Man.png" width="440" height="320"/>
</p>

#### 2. DCGAN vs LSGAN Inference(Epoch = 2, 4, 6, 8, 10)
1. DCGAN
<p align="center">
  <img src="./result/DCGAN_man.gif" width="350" height="260"/>
</p>

2. LSGAN
<p align="center">
  <img src="./result/LSGAN_man.gif" width="350" height="260"/>
</p>

