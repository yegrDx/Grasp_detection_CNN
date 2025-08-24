
Этот проект реализует **нейросетевой подход к захвату объектов** с использованием [Cornell Grasp Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp).  
Модель обучается предсказывать оптимальные точки захвата на изображениях глубины.  

---

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

---


Для обучения используется **Cornell Grasp Dataset**.  
Скачайте и распакуйте его в удобную директорию, например:

```
./datasets/cornell_dataset
```

---


Запустить обучение можно с помощью команды:

```bash
python train.py   --data <path_to_your_data>   --epochs <num_of_epochs>   --bs <batch_size>   --lr <learning_rate>   --out <path_to_your_weights>   --device <cuda/cpu>   --resume <path_to_old_weights_for_retraining(optional)>
```

### Пример:
```bash
python train.py   --data ./datasets/cornell_dataset   --epochs 30   --bs 16   --lr 1e-3   --out ./checkpoints   --device cuda
```

---

Для запуска инференса используйте команду:

```bash
python inference.py   --weights ./checkpoints/ggcnn_best.pt   --depth ./datasets/cornell_dataset/07/pcd0700d.tiff   --out grasp_vis.png   --device cpu
```

В результате будет сгенерировано изображение с визуализацией захвата.

---

