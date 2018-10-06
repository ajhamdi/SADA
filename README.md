# BLACK BOX GAN

please Matthias use the following comman to learn the paramters in the selfdriving folder 

```
python main.py --is_selfdrive=True --is_train=True --valid_size=50 --log_frq=10 --batch_size=64 --induced_size=50 --nb_paramters=3 --nb_steps=600  --learning_rate_t=0.0001 --learning_rate_g=0.0001
```
`valid_size` : is the number of paramters u will be generating eventually <br>
`nb_paramters` is the nuber of parameters per sample ... 3 here <br>
`nb_steps` : is the number of training steps of teh GAN <br>
`log_frq=10` : how often u save the weights of teh network<br>
`induced_size`: is the number of best samples that will be picked out of the total number u have in input.csv <br>
