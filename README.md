It supports [Cornell grasp dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp)
You can traine the net by command:
python train.py --data <path_to_your_data> --epochs <num_of_epochs> --bs <batch_size> --lr <learning_rate> --out <path_to_your_weights> --device <cuda/cpu> --resume <path_to_old_weights_for_retraing(optionally) 
Example:
'''python train.py --data cornell_dataset --epochs 30 --bs 16 --lr 1e-3 --out ./checkpoints --device cuda'''
Example for inference:
'''python inference.py --weights ./checkpoints/ggcnn_best.pt --depth cornell_dataset\07\pcd0700d.tiff --out grasp_vis.png --device cpu'''
