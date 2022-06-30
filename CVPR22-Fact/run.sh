python train.py -project fact -dataset cub200 -base_mode 'ft_cos'  -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.005 \
 -lr_new 0.1  -decay 0.0005 -epochs_base 100 -schedule Cosine -gpu 0,1 -temperature 16 -dataroot ./data -alpha 0.5 -balance 0.01 \
 -loss_iter 50 -eta 0.1 -batch_size_base 64 >> CUB_e100_b64_efficientNet.txt 

 python train.py -project fact -dataset cub200 -base_mode 'ft_cos'  -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.005 \
 -lr_new 0.1  -decay 0.0005 -epochs_base 100 -schedule Cosine -gpu 1,0 -temperature 16 -dataroot ./data -alpha 0.5 -balance 0.01 \
 -loss_iter 50 -eta 0.1 -batch_size_base 64 >> CUB_e100_b64_efficientNet_adamw.txt 