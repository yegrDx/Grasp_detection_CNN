
import os, argparse, math
import numpy as np
from PIL import Image
import torch
from ggcnn_model import GGCNN
from dataset_cornell import depth_png_to_meters

def postprocess(pred, depth_img):

    q = pred['pos'][0,0].cpu().numpy()
    cos = pred['cos'][0,0].cpu().numpy()
    sin = pred['sin'][0,0].cpu().numpy()
    w = pred['width'][0,0].cpu().numpy()
    yx = np.unravel_index(np.argmax(q), q.shape)
    y, x = int(yx[0]), int(yx[1])
    angle = 0.5 * math.atan2(sin[y,x], cos[y,x])  # radians
    width_px = float(w[y,x])
    quality = float(q[y,x])
    return {'y': y, 'x': x, 'angle': angle, 'width_px': width_px, 'quality': quality}

def save_vis(depth, result, out_path):
    import matplotlib.pyplot as plt
    H,W = depth.shape
    y,x = result['y'], result['x']
    ang = result['angle']
    w = result['width_px']
    # line endpoints
    dx = math.cos(ang) * w/2
    dy = math.sin(ang) * w/2

    plt.figure(figsize=(6,6))
    plt.imshow(depth, cmap='gray')
    plt.scatter([x],[y], s=30)
    plt.plot([x-dx, x+dx], [y-dy, y+dy])
    plt.title(f"Q={result['quality']:.3f}, angle={result['angle']:.2f}rad, w={w:.1f}px")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--depth', type=str, required=True, help='Path to depth PNG from Cornell (uint16 mm or 8-bit)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--size', type=int, default=300)
    p.add_argument('--out', type=str, default='inference_vis.png')
    args = p.parse_args()

    device = torch.device(args.device)
    model = GGCNN(input_channels=1).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()

    depth_img = Image.open(args.depth)
    depth = depth_png_to_meters(depth_img)

    H,W = depth.shape
    s = min(H,W)
    top = (H - s)//2; left = (W - s)//2
    crop = depth[top:top+s, left:left+s]
    crop = np.array(Image.fromarray(crop).resize((args.size,args.size), Image.NEAREST), dtype=np.float32)

    m = crop.mean(); sd = crop.std() + 1e-6
    crop_n = (crop - m) / sd
    x = torch.from_numpy(crop_n[None,None,...]).float().to(device)

    with torch.no_grad():
        pred = model(x)
    result = postprocess(pred, crop)

    print('Best grasp:', result)
    save_vis(crop, result, args.out)
    print('Saved vis to', args.out)

if __name__ == '__main__':
    main()
