<img src='.//examples/intro.png' align="right" width=500>
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
git clone https://github.com/ajhamdi/SADA
cd SADA
```

- Download the dataset that contains the 3D shapes and the environments from [this link](https://drive.google.com/drive/folders/1IFKOivjYXBQOhnc2WV7E4hipxCtSoB4u?usp=sharing) and place the folder in the same project dir with name `3d/training_pascal`. 


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
