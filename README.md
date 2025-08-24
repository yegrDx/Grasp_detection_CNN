This model is trained on [Cornell Grasp Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp).  
The model predicts the position, angle and width for grasping object.

---
Installation
```bash
git clone https://github.com/yegrDx/Grasp_detection_CNN.git
cd Grasp_detection_CNN
pip install -r requirements.txt
```

---



Command for training:

```bash
python train.py --data <path_to_your_data> --epochs <num_of_epochs> --bs <batch_size> --lr <learning_rate> --out <path_to_your_weights> --device <cuda/cpu> --resume <path_to_old_weights_for_retraining(optional)>
```

### Examle for train:
```bash
python train.py --data ./datasets/cornell_dataset --epochs 30 --bs 16 --lr 1e-3 --out ./checkpoints --device cuda
```

---

### Example for inference on one depth card

```bash
python inference.py --weights ./checkpoints/ggcnn_best.pt --depth ./datasets/cornell_dataset/07/pcd0700d.tiff --out grasp_vis.png --device cpu
```

---

