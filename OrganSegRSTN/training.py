import os
import sys
import time
import pandas as pd
from torch.autograd import Variable
from utils import parameters
from OrganSegRSTN.models import *
from torchvision.utils import save_image
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)




def train(DATA_PATH, CURRENT_FOLD, ORGAN_NUMBER, LOW_RANGE, HIGH_RANGE, SLICE_THRESHOLD, SLICE_THICKNESS, TRAINING_ORGAN_ID,
          X_Y_Z, TRAINING_GPU, LEARNING_RATE1, LEARNING_RATE_M1, LEARNING_RATE2, LEARNING_RATE_M2, TRAINING_MARGIN,
          TRAINING_PROB, TRAINING_SAMPLE_BATCH, TRAINING_EPOCH_S, TRAINING_EPOCH_I, TRAINING_EPOCH_J, LR_DECAY_EPOCH_J_STEP,
          TRAINING_LOG):

    from utils.utils_functions import snapshot_path,pretrained_model_path
    import torch

    data_path = DATA_PATH
    current_fold = CURRENT_FOLD
    organ_number = int(ORGAN_NUMBER)
    low_range = int(LOW_RANGE)
    high_range = int(HIGH_RANGE)
    slice_threshold = float(SLICE_THRESHOLD)
    slice_thickness = int(SLICE_THICKNESS)
    organ_ID = int(TRAINING_ORGAN_ID)
    plane = X_Y_Z
    GPU_ID = int(TRAINING_GPU)
    learning_rate1 = float(LEARNING_RATE1)
    learning_rate_m1 = int(LEARNING_RATE_M1)
    learning_rate2 = float(LEARNING_RATE2)
    learning_rate_m2 = int(LEARNING_RATE_M2)
    crop_margin = int(TRAINING_MARGIN)
    crop_prob = float(TRAINING_PROB)
    crop_sample_batch = int(TRAINING_SAMPLE_BATCH)
    snapshot_path = os.path.join(snapshot_path, 'SIJ_training_' + 'x' + str(learning_rate_m1) + ',' + str(crop_margin))
    epoch = 10

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    FCN_weights = os.path.join(pretrained_model_path, 'fcn8s_from_caffe.pth')
    if not os.path.isfile(FCN_weights):
        raise RuntimeError(
            'Please Download <http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU> from the Internet ...')

    from OrganSegRSTN.dataset import DataLayer
    training_set = DataLayer(data_path=data_path, current_fold=int(current_fold), organ_number=organ_number, \
                             low_range=low_range, high_range=high_range, slice_threshold=slice_threshold,
                             slice_thickness=slice_thickness, \
                             organ_ID=organ_ID, plane=plane)

    batch_size = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=1,
                                              drop_last=True)
    print(str(current_fold) + plane, len(trainloader))
    print(epoch)

    RSTN_model = FCN8s(3)


    model_parameters = filter(lambda p: p.requires_grad, RSTN_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters:', params)


    optimizer = torch.optim.Adam([
        {'params': [param for name, param in RSTN_model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * learning_rate1},
        {'params': [param for name, param in RSTN_model.named_parameters() if name[-4:] != 'bias'],
         'lr': learning_rate1, 'weight_decay': 1e-4}
    ], betas=(0.95, 0.999))

    criterion = DSC_loss()

    RSTN_model = RSTN_model.cuda()
    RSTN_model.train()

    RSTN_dict = RSTN_model.state_dict()
    pretrained_dict = torch.load(FCN_weights)
    # 1. filter out unnecessary keys
    pretrained_dict_coarse = {'coarse_model.' + k: v
                              for k, v in pretrained_dict.items()
                              if 'coarse_model.' + k in RSTN_dict and 'score' not in k}
    pretrained_dict_fine = {'fine_model.' + k: v
                            for k, v in pretrained_dict.items()
                            if 'fine_model.' + k in RSTN_dict and 'score' not in k}
    # 2. overwrite entries in the existing state dict
    RSTN_dict.update(pretrained_dict_coarse)
    RSTN_dict.update(pretrained_dict_fine)
    # 3. load the new state dict
    RSTN_model.load_state_dict(RSTN_dict)
    print( 'load pre-trained FCN8s model successfully!')

    try:
        for e in range(epoch):
            total_loss = 0.0
            start = time.time()
            for index, (image, label) in enumerate(trainloader):
                start_it = time.time()
                # 梯度清零
                optimizer.zero_grad()
                image, label = image.cuda().float(), label.cuda().float()
                image, label = Variable(image), Variable(label)
                outputs = RSTN_model(image)
                #result1 = np.array(image.cpu().detach().numpy())
                #np.savetxt('image.txt', result1[0][0])
                #result2 = np.array(outputs.cpu().detach().numpy())
                #np.savetxt('outputs.txt', result2[0][0])
                #print(image.cpu().detach().numpy())
                #print(outputs.cpu().detach().numpy())
                loss = criterion(outputs, label)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                print(str(current_fold) + plane ,
                      "Epoch[%d/%d], Iter[%05d],  Loss %.4f, Time Elapsed %.2fs" \
                      % (e + 1, epoch, index,  loss.item(),time.time() - start_it))
                del image, label, outputs, loss


            print(str(current_fold) + plane ,
                  "Epoch[%d], Total  Loss %.4f, Time elapsed %.2fs" \
                  % (e + 1, total_loss / len(trainloader), time.time() - start))

    except KeyboardInterrupt:
        print('!' * 10, 'save before quitting ...')
    finally:
        snapshot_name = 'FD' + str(current_fold) + '_' + \
                        plane  + str(slice_thickness) + '_' + str(organ_ID)
        RSTN_snapshot = os.path.join(snapshot_path, snapshot_name) + '.pkl'
        torch.save(RSTN_model.state_dict(), RSTN_snapshot)
        print('#' * 10, 'end of ' + str(current_fold) + plane  + ' training stage!')





