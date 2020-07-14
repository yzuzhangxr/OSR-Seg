import os
import sys
import time
from OrganSegRSTN.models import *
from torchvision.utils import save_image





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
    epoch = {}
    epoch['S'] = int(TRAINING_EPOCH_S)
    epoch['I'] = int(TRAINING_EPOCH_I)
    epoch['J'] = int(TRAINING_EPOCH_J)
    epoch['lr_decay'] = int(LR_DECAY_EPOCH_J_STEP)

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
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                              drop_last=True)
    print(str(current_fold) + plane, len(trainloader))
    print(epoch)
    getLogs = open('data/logs/logs.txt', 'a+')
    getLogs.write('hello'+'\n')
    getLogs.close()

    RSTN_model = RSTN(crop_margin=crop_margin, \
                      crop_prob=crop_prob, crop_sample_batch=crop_sample_batch)
    RSTN_snapshot = {}

    model_parameters = filter(lambda p: p.requires_grad, RSTN_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters:', params)

    optimizer = torch.optim.SGD(
        [
            {'params': get_parameters(RSTN_model, coarse=True, bias=False, parallel=False)},
            {'params': get_parameters(RSTN_model, coarse=True, bias=True, parallel=False),
             'lr': learning_rate1 * 2, 'weight_decay': 0},
            {'params': get_parameters(RSTN_model, coarse=False, bias=False, parallel=False),
             'lr': learning_rate1 * 5},
            {'params': get_parameters(RSTN_model, coarse=False, bias=True, parallel=False),
             'lr': learning_rate1 * 10, 'weight_decay': 0}
        ],
        lr=learning_rate1,
        momentum=0.99,
        weight_decay=0.0005)

    criterion = DSC_loss()
    COARSE_WEIGHT = 1 / 3

    RSTN_model = RSTN_model.cuda()
    RSTN_model.train()

    for mode in [ 'S','I', 'J']:
        if mode == 'S':
            '''
            在pytorch中，torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数，
            state_dict作为python的字典对象将每一层的参数映射成tensor张量，需要注意的是torch.nn.Module模块中的state_dict只包含卷积层和全连接层的参数，
            当网络中存在batchnorm时，例如vgg网络结构，torch.nn.Module模块中的state_dict也会存放batchnorm's running_mean，
            关于batchnorm详解可见https://blog.csdn.net/wzy_zju/article/details/81262453

            torch.optim模块中的Optimizer优化器对象也存在一个state_dict对象，此处的state_dict字典对象包含state和param_groups的字典对象，
            而param_groups key对应的value也是一个由学习率，动量等参数组成的一个字典对象。

            因为state_dict本质上Python字典对象，所以可以很好地进行保存、更新、修改和恢复操作（python字典结构的特性），从而为PyTorch模型和优化器增加了大量的模块化。

            '''










            print(plane + mode, 'load pre-trained FCN8s model successfully!')

        elif mode == 'I':
            print(plane + mode, 'load S model successfully!')
        elif mode == 'J':
            print(plane + mode, 'load I model successfully!')
        else:
            raise ValueError("wrong value of mode, should be in ['S', 'I', 'J']")

        try:
            for e in range(epoch[mode]):
                total_loss = 0.0
                total_coarse_loss = 0.0
                total_fine_loss = 0.0
                start = time.time()
                for index, (image, label) in enumerate(trainloader):
                    start_it = time.time()
                    # 梯度清零
                    optimizer.zero_grad()
                    image, label = image.cuda().float(), label.cuda().float()
                    coarse_prob, fine_prob = RSTN_model(image, label, mode=mode)
                    coarse_loss = criterion(coarse_prob, label)
                    fine_loss = criterion(fine_prob, label)
                    loss = COARSE_WEIGHT * coarse_loss + (1 - COARSE_WEIGHT) * fine_loss
                    total_loss += loss.item()
                    total_coarse_loss += coarse_loss.item()
                    total_fine_loss += fine_loss.item()
                    loss.backward()
                    optimizer.step()
                    #x = coarse_prob[0]
                    #y = fine_prob[0]
                    #z = label[0]
                    #img = torch.stack([x , y , z], 0)
                    #save_image(img.cpu(), os.path.join('./train_img', str(e) + '.png'))
                    print(str(current_fold) + plane + mode,
                          "Epoch[%d/%d], Iter[%05d], Coarse/Fine/Avg Loss %.4f/%.4f/%.4f, Time Elapsed %.2fs" \
                          % (e + 1, epoch[mode], index, coarse_loss.item(), fine_loss.item(), loss.item(),
                             time.time() - start_it))
                    del image, label, coarse_prob, fine_prob, loss, coarse_loss, fine_loss

                if mode == 'J' and (e + 1) % epoch['lr_decay'] == 0:
                    print('lr decay')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5

                print(str(current_fold) + plane + mode,
                      "Epoch[%d], Total Coarse/Fine/Avg Loss %.4f/%.4f/%.4f, Time elapsed %.2fs" \
                      % (e + 1, total_coarse_loss / len(trainloader), total_fine_loss / len(trainloader),
                         total_loss / len(trainloader), time.time() - start))
                getLogs = open('data/logs/logs.txt','a+')
                getLogs.write(str(current_fold) + plane + mode + ' ' + 'Epoch[' + str(e+1) + str(epoch[mode]) +
                              '], Total Coarse/Fine/Avg Loss' + ' ' + str(total_coarse_loss / len(trainloader)) + '/' +
                              str(total_fine_loss / len(trainloader)) + '/' + str(total_loss / len(trainloader)) +
                              ', Time elapsed' + str(time.time() - start) + 's')
                getLogs.write('\n')
                getLogs.close()
            #save_image(img.cpu(), os.path.join('./train_img', str(e) + '.png'))
        except KeyboardInterrupt:
            print('!' * 10, 'save before quitting ...')
        finally:
            snapshot_name = 'FD' + str(current_fold) + '_' + \
                            plane + mode + str(slice_thickness) + '_' + str(organ_ID)
            RSTN_snapshot[mode] = os.path.join(snapshot_path, snapshot_name) + '.pkl'
            torch.save(RSTN_model.state_dict(), RSTN_snapshot[mode])
            print('#' * 10, 'end of ' + str(current_fold) + plane + mode + ' training stage!')


