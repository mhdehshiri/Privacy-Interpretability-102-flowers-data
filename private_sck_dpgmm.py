

###Private EM Contrastive Learning
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches


import numpy as np
import PIL
## Standard libraries
import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim



# !pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 torchtext==0.14.1 fastai==2.7.11
# !pip install tokenizers
# !pip install torchdata==0.5.1

# Torchvision
import torchvision
from torchvision import transforms

import itertools
import tqdm
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
from sklearn.mixture import BayesianGaussianMixture
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

sns.set(rc={'figure.figsize':(11.7,8.27)})
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


import pickle
from torch.utils.data import (
    Dataset,
    DataLoader,
)

######initialization%%%%%%%

##address to model and data
epsilon = 1
# epsilon = 0.5
# epsilon = 0.05
Train_op = False
Test_op = True

if epsilon == 1 :
  model_open_name = 'train_model_supContrastive_Dpgmm_epsilon=1_DP_SCKDPGMM.pt'
  kmean_model_name = 'kmeans_10cluster_dpgmmepsilon=1_DP_SCKDPGMM.pickle'
  dpgmm_name = 'dpgmm0_01epsilon=1_DP_SCKDPGMM.pickle'

if epsilon == 0.5 :
  model_open_name = 'train_model_supContrastive_Dpgmm_epsilon=0.5_DP_SCKDPGMM.pt'
  kmean_model_name = 'kmeans_10cluster_dpgmmepsilon=0.5_DP_SCKDPGMM.pickle'
  dpgmm_name = 'dpgmm0_01epsilon=0.5_DP_SCKDPGMM.pickle'

if epsilon == 0.05 :
  model_open_name = 'train_model_supContrastive_Dpgmm_epsilon=0.05_DP_SCKDPGMM.pt'
  kmean_model_name = 'kmeans_10cluster_dpgmmepsilon=0.05_DP_SCKDPGMM.pickle'
  dpgmm_name = 'dpgmm0_01epsilon=0.05_DP_SCKDPGMM.pickle'

open_data_add , open_label_add = 'data/open_data79points' , 'data/open_label79points'
train_data_add , trian_label_add ='data/train_data','data/train_labels'
test_data_add , test_label_add = 'data/test_data'  , 'data/test_labels'



######END_initialization%%%%%

class CustomStarDataset(Dataset):
   def __init__(self, data_add , label_add , transform=None):

        with open(data_add + '.pickle' , 'rb') as f:
          self.data = np.array(pickle.load(f))
        with open(label_add + '.pickle' , 'rb') as f:
          self.labels = np.array(pickle.load(f))

        self.transform = transform

   def __len__(self):
        return len(self.data)

   def __getitem__(self, index):
        image = PIL.Image.fromarray(self.data[index , : , : , :])
        y_label = torch.tensor(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]



contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=128),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=0.5,
                                                                     contrast=0.5,
                                                                     saturation=0.4,
                                                                     hue=0.3)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.3),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])



open_data_contrast = CustomStarDataset(data_add=open_data_add,label_add = open_label_add , transform=ContrastiveTransformations(contrast_transforms, n_views=2))
train_data_contrast = CustomStarDataset(data_add=train_data_add,label_add = trian_label_add , transform=ContrastiveTransformations(contrast_transforms, n_views=2))
test_data_contrast = CustomStarDataset(data_add=test_data_add,label_add = test_label_add , transform=ContrastiveTransformations(contrast_transforms, n_views=2))


train_data_tensor = CustomStarDataset(data_add=train_data_add,label_add = trian_label_add , transform=transforms.ToTensor())
test_data_tensor = CustomStarDataset(data_add=test_data_add,label_add = test_label_add , transform=transforms.ToTensor())
open_data_tensor = CustomStarDataset(data_add=open_data_add,label_add = open_label_add , transform=transforms.ToTensor())

train_loader = DataLoader(train_data_contrast, batch_size=128, shuffle=False)
test_loader = DataLoader(test_data_contrast, batch_size=len(test_data_contrast), shuffle=False)
open_data_loader = DataLoader(dataset= open_data_contrast, batch_size=len(open_data_contrast), shuffle=True)


"""## Supervised_contrast

"""

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

"""##DPGMM"""

dpgmm = mixture.BayesianGaussianMixture(
    n_components=10,
    covariance_type="full",
    weight_concentration_prior=0.005,
    weight_concentration_prior_type="dirichlet_process",
    mean_precision_prior=1e-2,
    covariance_prior=1e-2 * np.eye(2),
    init_params="random",
    max_iter=5,
    warm_start=True,
    random_state=2,
)

"""## Model_initial

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.act = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(8192*4, 512)

        # Encoder head
        self.fc4 = nn.Linear(512, 128)
        # Projection head (See explanation in SimCLRv2)
        self.mlp1 = nn.Linear(128, 256)
        self.mlp2 = nn.Linear(256, 32)

    def forward(self, data1 , data2, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = data1
            augm_2 = data2

            x = self.conv1(augm_1)

            x = self.act(self.conv2(x))

            x = self.pool2(x)

            x = self.flat(x)

            x = self.fc3(x)

            h_1 = self.fc4(x)

            x = self.conv1(augm_2)

            x = self.act(self.conv2(x))

            x = self.pool2(x)

            x = self.flat(x)

            x = self.fc3(x)

            h_2 = self.fc4(x)

        else:

            augm_1 = data1

            augm_2 = data2

            x = self.conv1(augm_1)

            x = self.act(self.conv2(x))

            x = self.pool2(x)

            x = self.flat(x)

            x = self.fc3(x)

            h_1 = self.fc4(x)

            return h_1

        # Transformation for loss function
        compact_h_1 = self.mlp2(self.mlp1(h_1))
        compact_h_2 = self.mlp2(self.mlp1(h_2))
        return h_1, h_2, compact_h_1, compact_h_2

"""# Training"""

# !pip install pytorch-metric-learning -q



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)




if Train_op :
  from pytorch_metric_learning.losses import NTXentLoss, CircleLoss
  loss_func_uns = CircleLoss()
  loss_func_sup = SupConLoss(temperature=0.1)

  loss_func_uns = CircleLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
  print("training phase is started : ")
  def train():
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(train_loader)):
      data1 = data[0][0].to(device)
      data2 = data[0][1].to(device)
      labels= data[1].detach().to(device)
      bs = labels.shape[0]

      optimizer.zero_grad()
      # Get data representations
      h_1, h_2, compact_h_1, compact_h_2 = model(data1 , data2 , train =True)

      embeddings = torch.cat((compact_h_1, compact_h_2))
      embeddings = torch.nn.functional.normalize(embeddings,dim=1)
      f1, f2 = torch.split(embeddings, [bs, bs], dim=0)
      features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1).to(device)
      Label2 = torch.cat((labels , labels))
      loss = loss_func_uns(embeddings,Label2)
      loss.backward()
      total_loss += loss.item()
      optimizer.step()
    return total_loss / len(train_loader)



  for epoch in range(1,20):
    loss = train()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    scheduler.step()


  model = Model().to(device)
  model.eval()

  open_orginal = open_data_tensor.data.copy()
  open_orginal = open_orginal.reshape(open_orginal.shape[0] ,open_orginal.shape[-1],
                                      open_orginal.shape[1],open_orginal.shape[2] )

  open_label = open_data_tensor.labels

  data1 = torch.Tensor(open_orginal.copy()).to(device)
  model.eval()
  h = model(data1.to(device),data1.to(device) , train = False)
  h = h.cpu().detach()

  kmeans0_1 = MiniBatchKMeans(n_clusters=10, batch_size=len(open_label),
                            random_state=0)
  kmeans0_1 = KMeans(n_clusters=12, random_state=0, n_init="auto" , warm_start=True)
  clf2 = LinearDiscriminantAnalysis(n_components=2)

  kmeans0_1.partial_fit(h)
  labels0 = kmeans0_1.predict(h)
  h_embedded = clf2.fit_transform(h , labels0)

  labels= dpgmm.fit_predict(h_embedded)

  labels0 = kmeans0_1.predict(h)
  h_embedded = clf2.fit_transform(h , labels0)
  labels= dpgmm.predict(h_embedded)

  # ax2 = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1],hue=open_label,
  #                     alpha=0.5, palette="tab10" , sizes=(20, 200), hue_norm=(15, 7), legend="full")
  # ax2.set_title('True Clustring of Open Data Before SCKDPGMM')
  # plt.savefig('True_CLustring_open_data_beforeSUPTR.png')
  # ax1 = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1],hue=labels,
  #                     alpha=0.5, palette="tab10" , sizes=(20, 200), hue_norm=(15, 7), legend="full")
  # ax1.set_title('DPGMM Clustring of Open Data Before SCKDPGMM')


  """##DPGMM_MODEL_training"""

  model.train()
  total_loss = 0


  print("Sub trian is started : ")

  for itera in range(5):

    for epoch in range(5) :

      if (itera < 4 or itera == 70 )and epoch < 5 :
        for epoch2 in range(5):
          for _, data in enumerate(tqdm.tqdm(open_data_loader)):

            data1 = data[0][0].to(device)
            data2 = data[0][1].to(device)


            optimizer.zero_grad()
            # Get data representations
            h_1, h_2, compact_h_1, compact_h_2 = model(data1 , data2)
            # Prepare for loss
            embeddings = torch.cat((compact_h_1, compact_h_2))
            # The same index corresponds to a positive pair
            indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
            labels = torch.cat((indices, indices))
            loss = loss_func_uns(embeddings, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

      for _, data in enumerate(tqdm.tqdm(open_data_loader)):

        data1 = data[0][0].to(device)
        data2 = data[0][1].to(device)

        h = model(data1.to(device),data1.to(device), train=False).cpu().detach()
        labels0 = kmeans0_1.predict(h)
        h_embedded = clf2.transform(h)
        labels= torch.Tensor(dpgmm.predict(h_embedded))

        bs = labels.shape[0]
        optimizer.zero_grad()
        h_1, h_2, compact_h_1, compact_h_2 = model(data1 , data2 , train =True)
        embeddings = torch.cat((compact_h_1, compact_h_2))
        f1, f2 = torch.split(embeddings, [bs, bs], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1).to(device)
        loss = loss_func_sup(features, labels)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

      data1 = torch.Tensor(open_orginal.copy())
      h = model(data1.to(device),data1.to(device), train=False)
      h = h.cpu().detach()
      kmeans0_1.partial_fit(h)
      labels0 = kmeans0_1.predict(h)
      h_embedded = clf2.fit_transform(h , labels0)
      dpgmm.fit(h_embedded)
      count , uni = np.unique(dpgmm.predict(h_embedded) , return_counts=True)
      print("unique numbers and counts and itera are : " , count , uni , itera*epoch )




  # torch.save( model.state_dict() , 'train_model_supContrastive_Dpgmm.pt')
  # pickle.dump(kmeans0_1, open('kmeans_10cluster_dpgmm.pickle', "wb"))
  # pickle.dump(dpgmm, open('dpgmm0_01.pickle', "wb"))



"""#Test """

if Test_op :
  model = Model().to(device)
  model.load_state_dict(torch.load(model_open_name))
  kmeans0_1 = pickle.load(open(kmean_model_name, 'rb'))
  dpgmm = pickle.load(open(dpgmm_name, 'rb'))
  clf2 = LinearDiscriminantAnalysis(n_components=2)
  model.eval()
  open_orginal = open_data_tensor.data.copy()
  open_orginal = open_orginal.reshape(open_orginal.shape[0] ,open_orginal.shape[-1],
                                      open_orginal.shape[1],open_orginal.shape[2] )
  data1 = torch.Tensor(open_orginal.copy())
  h = model(data1.to(device),data1.to(device), train=False)
  h = h.cpu().detach()
  labels0 = kmeans0_1.predict(h)
  h_embedded = clf2.fit_transform(h , labels0)
  labels = dpgmm.predict(h_embedded)

  # ax1 = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1],hue=labels,
  #                     alpha=0.5, palette="tab10" , sizes=(20, 200), hue_norm=(15, 7), legend="full")
  # ax1.set_title('DPGMM Clustring of Open Data After SCKDPGMM')

  open_label = open_data_tensor.labels

  if epsilon == 1 :
    open_label[np.where(open_label==49) ] = 1
    open_label[np.where(open_label==15) ] = 0
    open_label[np.where(open_label==42) ] = 2

  if epsilon == 0.5 :
    open_label[np.where(open_label==49) ] = 1
    open_label[np.where(open_label==15) ] = 0
    open_label[np.where(open_label==42) ] = 2
#102/
  if epsilon == 0.05 :
    open_label[np.where(open_label==49) ] = 0
    open_label[np.where(open_label==15) ] = 1
    open_label[np.where(open_label==42) ] = 2

  print("Test operation is done for epsilon = " , epsilon)
  diff = abs(labels - open_label)
  c_nonz = np.count_nonzero(diff)
  acc_epsion = (79-c_nonz)/79
  print('Accuracy is : ' ,acc_epsion)
  # ax2 = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1],hue=open_label,
  #                     alpha=0.5, palette="tab10" , sizes=(20, 200), hue_norm=(15, 7), legend="full")
  # ax2.set_title('True Clustring of Open Data After SCKDPGMM')



