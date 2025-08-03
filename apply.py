import numpy as np
import torch
from utils import *
from model import UNet
import os
import time
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    num_GPU = torch.cuda.device_count()
    print(f"Number of GPUs: {num_GPU}")

output_dir = './prediction'
os.makedirs(output_dir, exist_ok=True)
 
def main():
    load_model()
    go_pred()

def load_model():
    global model
    checkpoint_path = f'./checkpoints/UNet_meander.pth'
    # checkpoint_path = f'./checkpoints/UNet_meander.pth'
    # checkpoint_path = f'./checkpoints/UNet_meander.pth'
    model = UNet(in_dim=1, out_dim=3, num_filters=32)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    print(f"Checkpoint path: {checkpoint_path}")

def go_pred():
    process_large_file(
        file_path='./fieldseismic/Parihaka_PSTM_cut_530_730_256 (Meandering channels).dat',
        n3=530, n2=730, n1=256, m1=224, m2=224, m3=224
    )
    # process_large_file(
    #     file_path='./fieldseismic/NW_441_901_201 (Tributary channel network).dat',
    #     n3=441, n2=901, n1=201, m1=200, m2=200, m3=200
    # )
    # process_large_file(
    #     file_path='./fieldseismic/Parihaka_PSTM_cut_224_224_224 (Submarine canyon).dat',
    #     n3=224, n2=224, n1=224, m1=176, m2=176, m3=176
    # )

def process_large_file(file_path, n3, n2, n1, m1, m2, m3):
    gx = np.fromfile(file_path, dtype='<f').reshape((n3, n2, n1)).transpose()
    fault_pred, river_pred, combined_pred = predict_with_rotations(m1, m2, m3, gx)

    file_name = os.path.basename(file_path)
    fault_path = os.path.join(output_dir, file_name.replace('.dat', '_fault_pred.dat'))  # Fault prediction.
    river_path = os.path.join(output_dir, file_name.replace('.dat', '_river_pred.dat'))  # Channel prediction.

    fault_pred.transpose().tofile(fault_path)
    river_pred.transpose().tofile(river_path)

    print(f"Saved predictions for {file_name}: {fault_path}, {river_path}")

def predict_with_rotations(m1, m2, m3, gx, num_rotations=3):
    fault_prob, river_prob, combined_prob = go_predict_subs(m1, m2, m3, gx)

    for i in range(1, num_rotations + 1):
        rotated_data = rotate_data(gx, k=i, axes=(1, 2))
        rotated_fault, rotated_river, rotated_combined = go_predict_subs(m1, m2, m3, rotated_data)

        fault_prob += rotate_data(rotated_fault, k=-i, axes=(1, 2))
        river_prob += rotate_data(rotated_river, k=-i, axes=(1, 2))
        combined_prob += rotate_data(rotated_combined, k=-i, axes=(1, 2))

    fault_prob /= (num_rotations + 1)
    river_prob /= (num_rotations + 1)
    combined_prob /= (num_rotations + 1)

    return fault_prob, river_prob, combined_prob

def go_predict_subs(m1, m2, m3, gx):
    n1, n2, n3 = gx.shape
    p1, p2, p3 = 16, 16, 16

    fault_prob = np.zeros((n1, n2, n3), dtype=np.single)
    river_prob = np.zeros((n1, n2, n3), dtype=np.single)
    combined_prob = np.zeros((n1, n2, n3), dtype=np.single)

    c1 = 1 + int(np.ceil((n1 - m1) / (m1 - p1)))
    c2 = 1 + int(np.ceil((n2 - m2) / (m2 - p2)))
    c3 = 1 + int(np.ceil((n3 - m3) / (m3 - p3)))

    for k3 in range(c3):
        for k2 in range(c2):
            for k1 in range(c1):
                b1, b2, b3 = k1 * (m1 - p1), k2 * (m2 - p2), k3 * (m3 - p3)
                e1, e2, e3 = min(b1 + m1, n1), min(b2 + m2, n2), min(b3 + m3, n3)
                b1, b2, b3 = max(0, e1 - m1), max(0, e2 - m2), max(0, e3 - m3)

                gk = gx[b1:e1, b2:e2, b3:e3]
                gk = dataprocess(gk)
                input_tensor = torch.from_numpy(gk).float().unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()

                fault_prob[b1:e1, b2:e2, b3:e3] = np.maximum(fault_prob[b1:e1, b2:e2, b3:e3], probs[0, 1, :, :, :])
                river_prob[b1:e1, b2:e2, b3:e3] = np.maximum(river_prob[b1:e1, b2:e2, b3:e3], probs[0, 2, :, :, :])
                combined_prob[b1:e1, b2:e2, b3:e3] = np.maximum(
                    combined_prob[b1:e1, b2:e2, b3:e3], probs[0, 1, :, :, :] + probs[0, 2, :, :, :]
                )

    return fault_prob, river_prob, combined_prob

def rotate_data(data, k=1, axes=(1, 2)):
    return np.rot90(data, k=k, axes=axes)

def dataprocess(sx):
    sxmean = np.mean(sx)
    sxstd = np.std(sx)
    return (sx - sxmean) / sxstd

if __name__ == '__main__':
    main()
