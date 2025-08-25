
from typing import Tuple, Dict, List
import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import math

def read_grasp_file(txt_path: str) -> np.ndarray:
   
    rects = []
    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
   
    for i in range(0, len(lines), 4):
        pts = []
        for j in range(4):
            x, y = lines[i+j].split()
            pts.append([float(x), float(y)])
        rects.append(np.array(pts))  # shape (4,2)
    return np.stack(rects, axis=0) if rects else np.zeros((0,4,2), dtype=np.float32)

def depth_png_to_meters(img: Image.Image) -> np.ndarray:
    a = np.array(img).astype(np.float32)
    # If uint16 in millimeters -> meters
    if a.dtype == np.uint16 or a.max() > 255:
        return a / 1000.0
    
    a = a / 255.0
    return a

def crop_square(arr: np.ndarray, size: int, top: int, left: int) -> np.ndarray:
    h, w = arr.shape[:2]
    t = max(0, min(top, h - size))
    l = max(0, min(left, w - size))
    return arr[t:t+size, l:l+size]

def rotate_point(cx, cy, x, y, angle):
    
    s, c = math.sin(angle), math.cos(angle)
    x -= cx; y -= cy
    x_new = x*c - y*s
    y_new = x*s + y*c
    return x_new + cx, y_new + cy

def polygon_to_mask(poly: np.ndarray, shape: Tuple[int,int]) -> np.ndarray:
    from skimage.draw import polygon as poly_draw
    r = poly[:,1]; c = poly[:,0]
    rr, cc = poly_draw(r, c, shape)
    mask = np.zeros(shape, dtype=np.float32)
    mask[rr, cc] = 1.0
    return mask

class CornellGraspDataset(Dataset):
    def __init__(self, root: str, split: Tuple[float,float]=(0.0,1.0), output_size: int = 300,
                 random_rotate: bool = True, random_zoom: bool = True, include_rgb: bool = False, seed: int = 42):
        super().__init__()
        self.root = root
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.include_rgb = include_rgb
        random.seed(seed)

        
        depth_files = sorted(glob.glob(os.path.join(root, "*", "*d.tiff")))
        self.items = []
        for d in depth_files:
            base = os.path.splitext(d)[0]  
            txt = base[:-1] + "cpos.txt" if base.endswith("d") else base + "cpos.txt"
            if not os.path.exists(txt):
                continue

            # rgb (если нужно)
            rgb = None
            cand_rgb = base[:-1] + "r.png"
            if os.path.exists(cand_rgb):
                rgb = cand_rgb

            self.items.append((d, txt, rgb))

        n = len(self.items)
        i0 = int(split[0]*n); i1 = int(split[1]*n)
        self.items = self.items[i0:i1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        depth_path, grasp_path, rgb_path = self.items[idx]
        depth = Image.open(depth_path)
        depth = depth_png_to_meters(depth)
        H, W = depth.shape
        
        rects = read_grasp_file(grasp_path)  
        if rects.shape[0] == 0:
            cx, cy = W/2, H/2
        else:
            cx = rects[:,:,0].mean()
            cy = rects[:,:,1].mean()
        
        angle = random.uniform(-math.pi/2, math.pi/2) if self.random_rotate else 0.0
        zoom = 1.0 + random.uniform(-0.2, 0.2) if self.random_zoom else 1.0
        size = int(self.output_size / zoom)
        top = int(max(0, cy - size/2))
        left = int(max(0, cx - size/2))
        d_crop = crop_square(depth, size, top, left)
        d_crop = np.array(Image.fromarray(d_crop).resize((self.output_size, self.output_size), Image.NEAREST))
        
        d_img = Image.fromarray(d_crop)
        d_img = d_img.rotate(angle * 180.0 / math.pi, resample=Image.NEAREST)
        d = np.array(d_img, dtype=np.float32)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        
        Q = np.zeros_like(d, dtype=np.float32)
        COS = np.zeros_like(d, dtype=np.float32)
        SIN = np.zeros_like(d, dtype=np.float32)
        W = np.zeros_like(d, dtype=np.float32)
        
        for rect in rects:
            
            rc = rect.copy()
            rc[:,0] -= left; rc[:,1] -= top
            
            sx = self.output_size / size; sy = self.output_size / size
            rc[:,0] *= sx; rc[:,1] *= sy
            
            cx_out, cy_out = self.output_size/2, self.output_size/2
            for k in range(4):
                rc[k,0], rc[k,1] = rotate_point(cx_out, cy_out, rc[k,0], rc[k,1], angle)
            
            p0, p1, p2, p3 = rc
            
            gcx = rc[:,0].mean(); gcy = rc[:,1].mean()
           
            v = ((p1+p2)/2) - ((p0+p3)/2)
            theta = math.atan2(v[1], v[0])  # radians
         
            w_vec = ((p0+p1)/2) - ((p2+p3)/2)
            width_px = np.linalg.norm(w_vec)
            
            mask = polygon_to_mask(rc, (self.output_size, self.output_size))
            Q[mask > 0.5] = 1.0
            COS[mask > 0.5] = math.cos(2*theta)
            SIN[mask > 0.5] = math.sin(2*theta)
            W[mask > 0.5] = width_px

       
        m = np.mean(d); s = np.std(d) + 1e-6
        d = (d - m) / s
        d = d[None, ...]  

        sample = {
            'x': torch.from_numpy(d).float(),
            'y': {
                'pos': torch.from_numpy(Q[None,...]).float(),
                'cos': torch.from_numpy(COS[None,...]).float(),
                'sin': torch.from_numpy(SIN[None,...]).float(),
                'width': torch.from_numpy(W[None,...]).float(),
            }
        }
        return sample
