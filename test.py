# from torch_utils import misc

# a = "layer1"
# a_list = []
# a_list.append(a)
# print(a_list)
# b_list = a_list.clone()
# # check if a in b_list has the same memory address
# print(b_list)
# import numpy as np
# import torch
# from torchvision.datasets import CIFAR100
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from score.inception import InceptionV3  # Adjust import as needed

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # 1. Load dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
# dataset = CIFAR100(root='./', train=True, download=True, transform=transform)
# loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

# # 2. Load InceptionV3
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
# model = InceptionV3([block_idx]).to(device)
# model.eval()

# # 3. Extract features
# features = []
# with torch.no_grad():
#     for imgs, _ in loader:
#         imgs = imgs.to(device)
#         pred = model(imgs)[0].view(imgs.size(0), -1)
#         features.append(pred.cpu().numpy())
# features = np.concatenate(features, axis=0)

# # 4. Compute mu and sigma
# mu = np.mean(features, axis=0)
# sigma = np.cov(features, rowvar=False)

# # 5. Save
# np.savez('./cifar100.train.npz', mu=mu, sigma=sigma)

import numpy as np
import torch

f = np.load('./stats/cifar100lt.train.npz')
print(f['mu'].shape, f['sigma'].shape)