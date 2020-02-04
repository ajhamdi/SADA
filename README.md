<img src='.//results/teapot.jpg' align="right" width=500>
<img src='./some_examples/robustness_out.gif' align="center" width=250>  
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

## Examples of Results
- ### visualizing the Deep networks average semantic profiles (1D) for 10 objects for 10 classes.
<img src='./results/toilet.jpg' align="center" width=500>

- ### Detecting robust regions of the networks with bounds-optimzing algorithms (1D).
<p float="left">
<img src='./some_examples/optimization1_video.gif' width=480 /> 
<img src='./results/run_120.jpg' width=380 />
</p>

- ### visualizing the Deep networks semantic profiles (2D) for 100 objects from 10 classes.
<p float="left">
<img src='./some_examples//myfull.gif' width=300>          
<img src='./results/bathtub2map.png' width=480>
</p>

- ### Detecting robust regions of the networks with bounds-optimzing algorithms (2D).
<img src='./results/bathtub2D.png' align="center" width=500>


- ### Visualizinfg the semantic bias in [ImageNet](http://www.image-net.org/) dataset for 10 differnt classes.
<img src='./results/rifle.png' align="center" width=500>


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
conda activate activate sada
```
- Clone this repo:
```bash
git clone https://github.com/ajhamdi/semantic-robustness
cd semantic-robustness
```


### Simple Colab Tutorial with a toy example:
We provide a simple tutorial on colab [here](https://colab.research.google.com/drive/1cZzTPu1uwftnRLqtIIjjqw-YZSKh4QYn) to test a toy example on some 3D objects and apply the bound optimzation algorithms for sample points in the 1D case directly on the cloud. The complete results obtained in the `results` directory are obtained as of the following sections  


### Dataset
- We collect 100 3D shapes from 10 classes from [ShapeNet](https://www.shapenet.org/) that are also exist in [ImagNet](http://www.image-net.org/) and made sure that networks trained on ImageNEt identifies these shapes of ShapeNet nefore proceeding. All the obj files are availabe in the `sacale` directory which contain the dataset. The classes are the following 
1. **aeroplane** : with ImageNet class label 404
1. **bathtub** : with ImageNet class label 435
1. **bench** : with ImageNet class label 703
1. **bottle** : with ImageNet class label 898
1. **chair** : with ImageNet class label 559
1. **cup** : with ImageNet class label 968
1. **piano** : with ImageNet class label 579
1. **rifle** : with ImageNet class label 413
1. **vase** : with ImageNet class label 883
1. **toilet** : with ImageNet class label 861
#### visualization:
<img src='./results/class_0_.gif' width=150>  <img src='./results/class_1_.gif' width=150>  <img src='./results/class_2_.gif' width=150>  <img src='./results/class_3_.gif' width=150>  <img src='./results/class_4_.gif' width=150>  <img src='./results/class_5_.gif' width=150> <img src='./results/class_6_.gif' width=150>  <img src='./results/class_7_.gif' width=150> <img src='./results/class_8_.gif' width=150>  <img src='./results/class_9_.gif' width=150> 

<br>
# BLACK BOX GAN

please Matthias use the following comman to learn the paramters in the selfdriving folder 

```
python main.py --is_selfdrive=True --is_train=True --valid_size=50 --log_frq=10 --batch_size=64 --induced_size=50 --nb_paramters=3 --nb_steps=600  --learning_rate_t=0.0001 --learning_rate_g=0.0001
```
* `valid_size` : is the number of paramters u will be generating eventually <br>
* `nb_paramters` is the nuber of parameters per sample ... 3 here <br>
* `nb_steps` : is the number of training steps of teh GAN <br>
* `log_frq=10` : how often u save the weights of teh network<br>
* `induced_size`: is the number of best samples that will be picked out of the total number u have in input.csv <br>
<br><br>
## where do play in the code ?
U will find the function `learn_selfdrive()` part of teh main class has your part ... please play only here to make our code exclusive 
<br>
## how to load the trained model and use the discrminator score ?
I just added that as `self.cont_train` option , in your `learn_selfdrive()` function , so just use the following command .. it will used the latest training model and randomly sample `FLAGS.K * FLAGS. indiced_size ` sample and sort them based on discrminator score , pick the best and worst `Induced_size` paramters and store them in the csv files in the self_drive folder.
<br>
use the following command please.. I have not tested it but it should work !  
```
python main.py --is_selfdrive=True --is_train=False --cont_train=True --valid_size=50 --log_frq=10 --batch_size=64 --induced_size=50 --nb_paramters=3 --nb_steps=600  --learning_rate_t=0.0001 --learning_rate_g=0.0001
```
