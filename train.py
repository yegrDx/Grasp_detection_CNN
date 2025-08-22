# train.py
import os, argparse, time
import torch
from torch.utils.data import DataLoader, random_split
from ggcnn_model import GGCNN, ggcnn_loss
from dataset_cornell import CornellGraspDataset

def seed_all(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--epochs', type=int, default=30, help='Additional epochs to train (if resuming)')
    p.add_argument('--bs', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', type=str, default='checkpoints')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--size', type=int, default=300)
    p.add_argument('--resume', type=str, default=None, help='Path to checkpoint (.pt) to resume training')
    args = p.parse_args()
    seed_all(123)

    os.makedirs(args.out, exist_ok=True)

    # --- data ---
    ds = CornellGraspDataset(args.data, split=(0.0, 1.0), output_size=args.size)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    ds_tr, ds_val = random_split(ds, [n_train, n_val])

    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True,  num_workers=args.workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    # --- device & model ---
    req = args.device.lower()
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if (req == 'cuda' and has_cuda) else 'cpu')
    if req == 'cuda' and not has_cuda:
        print("[WARN] CUDA недоступна, обучаю на CPU.")

    model = GGCNN(input_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- resume (после инициализации device/model/opt) ---
    start_epoch = 0
    best_val = float('inf')
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state)
        if isinstance(ckpt, dict):
            start_epoch = int(ckpt.get('epoch', 0))
            best_val = float(ckpt.get('val_loss', best_val))
            # если сохранял optimizer/scaler — можно подгрузить:
            if 'optimizer' in ckpt:
                try:
                    opt.load_state_dict(ckpt['optimizer'])
                except Exception:
                    print("[WARN] Не удалось восстановить optimizer.state_dict(), продолжаю с новым оптимизатором.")
            if 'scaler' in ckpt and scaler is not None:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                except Exception:
                    print("[WARN] Не удалось восстановить scaler.state_dict().")
        print(f"[INFO] Loaded epoch={start_epoch}, best_val={best_val:.4f}")

    # --- training loop ---
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0
        for batch in dl_tr:
            x = batch['x'].to(device, non_blocking=True)
            y = {k: v.to(device, non_blocking=True) for k, v in batch['y'].items()}

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    losses = ggcnn_loss(pred, y)
                    loss = losses['loss']
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(x)
                losses = ggcnn_loss(pred, y)
                loss = losses['loss']
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            tr_loss += loss.item() * x.size(0)
        tr_loss /= max(1, n_train)

        # --- validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dl_val:
                x = batch['x'].to(device, non_blocking=True)
                y = {k: v.to(device, non_blocking=True) for k, v in batch['y'].items()}
                pred = model(x)
                losses = ggcnn_loss(pred, y)
                val_loss += losses['loss'].item() * x.size(0)
        val_loss /= max(1, n_val)

        dt = time.time() - t0
        print(f'Epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f} | {dt:.1f}s')

        # save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.out, 'ggcnn_best.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'scaler': (scaler.state_dict() if scaler is not None else None),
                'val_loss': best_val,
                'epoch': epoch
            }, ckpt_path)
            print('Saved', ckpt_path)

    # final
    last_path = os.path.join(args.out, 'ggcnn_last.pt')
    torch.save({'model': model.state_dict(), 'epoch': epoch}, last_path)
    print('Saved', last_path)

if __name__ == '__main__':
    main()
