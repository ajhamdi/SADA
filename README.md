<img src='.//examples/intro.png' align="right" width=500>
<br><br><br><br>

# SADA: Semantic Adversarial Diagnostic Attacks for Autonomous Applications
### [Paper](https://arxiv.org/pdf/1812.02132.pdf) |  [video](https://youtu.be/clguL24kVG0)  <br>
Tensorflow implementation of [the paper](https://arxiv.org/abs/1812.02132) in [AAAI 2020](https://aaai.org/Conferences/AAAI-20/). The paper tries to address the robustness of Deep Neeural Networks, but not from pixel-level perturbation lense, rather from semantic lense in which the perturbation happens in the latent parameters that generate the image. This type of robustness is important for safety-critical applications like self-driving cars in which tolerance of error is very low and risk of failure is high. <br><br>
[SADA: Semantic Adversarial Diagnostic Attacks for Autonomous Applications](https://arxiv.org/pdf/1812.02132.pdf)  
 [Abdullah Hamdi](https://abdullahamdi.com/), [Matthias Muller](https://matthias.pw/), [Bernard Ghanem](http://www.bernardghanem.com/)

## Citation

If you find this useful for your research, please use the following.

```
@article{DBLP:journals/corr/abs-1812-02132,
  author    = {Abdullah Hamdi and
               Matthias M{\"{u}}ller and
               Bernard Ghanem},
  title     = {{SADA:} Semantic Adversarial Diagnostic Attacks for Autonomous Applications},
  journal   = {CoRR},
  volume    = {abs/1812.02132},
  year      = {2018},
  url       = {http://arxiv.org/abs/1812.02132},
  archivePrefix = {arXiv},
  eprint    = {1812.02132},
  timestamp = {Fri, 15 Nov 2019 17:16:40 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1812-02132},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Prerequisites
- Linux 
- Python 2 or 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN
- [Blender 2.79](https://www.blender.org/download/releases/2-79/)

## Getting Started
### Installation
- install [Blender](https://www.blender.org/download/releases/2-79/) with the version `blender-2.79b-linux-glibc219-x86_64` and add it to your `PATH` by adding the command `export PATH="${PATH}:/home/PATH/TO/blender-2.79b-linux-glibc219-x86_64"` in `/home/.bashrc` file . make sure at the end that you can run `blender` command from your shell script. 

- install the following `conda` environment as follows: 
```bash
conda env create -f environment.yaml
conda  activate sada
```
- Clone this repo:
```bash
git clone https://github.com/ajhamdi/SADA
cd SADA
```

- Download the dataset that contains the 3D shapes and the environments from [this link](https://drive.google.com/drive/folders/1IFKOivjYXBQOhnc2WV7E4hipxCtSoB4u?usp=sharing) and place the folder in the same project dir with name `3d/training_pascal`. 

- Download the weights for YOLOv3 from [this link](https://drive.google.com/file/d/1FeHobYulruf98ZOnWVpmqK8vza2u6MX-/view?usp=sharing) and place in the `detectos` dir. 

<br><br>

### Dataset
- We collect 100 3D shapes from 10 classes from [ShapeNet](https://www.shapenet.org/) and Pascal3D . All the sahpes are available inside the blender environment `3d/training_pascal/training.blend` file. The classes are the following 
1. **aeroplane** 
1. **bench** 
1. **bicycle**
1. **boat**
1. **bottle** 
1. **bus** 
1. **car** 
1. **chair** 
1. **dining table** 
1. **motorbike** 
1. **train** 
1. **truck** 

- The parameters that control the environment are 8 as follows  
1. **camera distance to the object** 
1. **camera azimuth angle** 
1. **camera pitch angle** 
1. **light source azimuth angle** 
1. **light source pitch angle** 
1. **color of the object (R-channel)** 
1. **color of the object (G-channel)** 
1. **color of the object (B-channel)** 


<br>

## Generating images from the 3D environment for a specific class with random parameters and storing the 2D dataset in the folder `generated` 

```
python main.py --is_gendist=True --class_nb= 0 --dataset_nb= 0 --gendist_size= 10000
```
* `is_gendist` : is the option to generate distribution of parameters and images  <br>
* `class_nb` the class of the 12 classes above to generate  <br>
* `dataset_nb` : is the number assigned to the dataset generated  <br>
* `gendist_size` : the number of inages generated  <br>
<br><br>


## training BBGAN
 

```
python main.py --is_train=True --valid_size=50 --log_frq=10 --batch_size=32 --induced_size=50 --nb_steps=600  --learning_rate_t=0.0001 --learning_rate_g=0.0001
```
* `class_nb` the class of the 12 classes above to generate  <br>
* `dataset_nb` : is the number assigned to the dataset generated  <br>
* `nb_steps` : is the number of training steps of the GAN <br>
* `log_frq=10` : how often u save the weights of the network<br>
* `induced_size`: is the number of best samples that will be picked out of the total numberof generated images  <br>
* `learning_rate_g`: the learning rate forf the generator
* `learning_rate_t`: the learning rate forf the discrminator
* `valid_size` : is the number of paramters u will be generating eventually for evaluation of the BBGAN <br>


<br><br>
