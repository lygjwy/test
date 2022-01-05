#!/bin/env bash

# execute command in parallel
# classify task
# python main.py --task classify --mode normal --arch resnet18 --batch_size 128 --epochs 200 --loss_fn mse --reduction sum --optimizer sgd --output_dir classify-resnet18 --gpu_idx 0 &> ./classify-resnet18/console.txt &
# python main.py --task classify --mode normal --arch resnet18_ae --batch_size 128 --epochs 200 --loss_fn mse --reduction sum --optimizer sgd --output_dir classify-resnet18_ae --gpu_idx 2 &> ./classify-resnet18_ae/console.txt &
# python main.py --task reconstruct --mode normal --arch ae --batch_size 128 --epochs 200 --loss_fn mse --reduction sum --optimizer adam --output_dir reconstruct-ae --gpu_idx 1 &> ./reconstruct-ae/console.txt &
# python main.py --task reconstruct --mode normal --arch resnet18_ae --batch_size 128 --epochs 200 --loss_fn mse --reduction sum --optimizer adam --output_dir reconstruct-resnet18_ae --gpu_idx 1 &> ./reconstruct-resnet18_ae/console.txt &
python main.py --task reconstruct --mode corrupt --arch ae --output_dir corrupt-ae --gpu_idx 2 &> ./corrupt-ae/console.txt &
python main.py --task reconstruct --mode corrupt --arch resnet18_ae --output_dir corrupt-resnet18_ae --gpu_idx 2 &> ./corrupt-resnet18_ae/console.txt &

python main.py --task classify --mode normal --arch resnet18 --preprocess normalize --optimizer sgd --output_dir normalize_classify-resnet18 --gpu_idx 1 &> ./normalize_classify-resnet18/console.txt &
python main.py --task classify --mode normal --arch resnet18 --preprocess normalize --optimizer sgd --output_dir normalize_classify-resnet18_ae --gpu_idx 0 &> ./normalize_classify-resnet18_ae/console.txt &