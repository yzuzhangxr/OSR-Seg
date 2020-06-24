from utils import parameters
import os
import time
from OrganSegRSTN.models import *
import numpy as np
from utils.utils_functions import *



data_path = parameters.DATA_PATH
current_fold = parameters.CURRENT_FLOD
organ_number = parameters.ORGAN_NUMBER
low_range = parameters.LOW_RANGE
high_range = parameters.HIGH_RANGE
slice_threshold = parameters.SLICE_THRESHOLD
slice_thickness = parameters.SLICE_THICKNESS
organ_ID = parameters.ORGAN_ID
GPU_ID = parameters.GPU_ID
learning_rate1 = parameters.LEARNING_RATE_1
learinge_rate_m1 = parameters.LEARNING_RATE_m1
learning_rate2  = parameters.LEARNING_RATE_2
learning_rate_m2 = parameters.LEARNING_RATE_m2
margin = parameters.MARGIN
prob = parameters.PROB
sample_batch = parameters.SAMPLE_BATCH
training_epcoh_s = parameters.TRAINING_EPCOH_S
training_epcoh_i = parameters.TRAINING_EPCOH_I
training_epcon_j = parameters.TRAINING_EPCOH_J
lr_decay_epoch_j_step = parameters.LR_DECAY_EPOCH_J_STEP


def coarse_testing(DATA_PATH, CURRENT_FOLD, ORGAN_NUMBER, LOW_RANGE, HIGH_RANGE, SLICE_THRESHOLD, SLICE_THICKNESS, TRAINING_ORGAN_ID,
                    X_Y_Z, TRAINING_GPU, LEARNING_RATE1, LEARNING_RATE_M1, LEARNING_RATE2, LEARNING_RATE_M2, TRAINING_MARGIN,
                    TRAINING_PROB, TRAINING_SAMPLE_BATCH, TRAINING_EPOCH_S, TRAINING_EPOCH_I, TRAINING_EPOCH_J, LR_DECAY_EPOCH_J_STEP):

    from utils.utils_functions import snapshot_path,result_path


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
    result_path = os.path.join(result_path, 'coarse_testing_' + 'x' + str(learning_rate_m1) + ',' + str(crop_margin))
    epoch = 'e' + str(TRAINING_EPOCH_S) + str(TRAINING_EPOCH_I) + str(TRAINING_EPOCH_J) + str(LR_DECAY_EPOCH_J_STEP)
    epoch_list = [epoch]
    snapshot_name = snapshot_name_from_timestamp(snapshot_path,current_fold, plane, 'J', slice_thickness, organ_ID)
    if snapshot_name == '':
        exit('Error: no valid snapshot directories are detected!')
    snapshot_directory = os.path.join(snapshot_path, snapshot_name)
    print('Snapshot directory: ' + snapshot_directory + ' .')
    snapshot = [snapshot_directory]
    print(str(len(snapshot)) + ' snapshots are to be evaluated.')
    for t in range(len(snapshot)):
        print('  Snapshot #' + str(t + 1) + ': ' + snapshot[t] + ' .')
    result_name = snapshot_name

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()
    while volume_list[len(volume_list) - 1] == '':
        volume_list.pop()
    DSC = np.zeros((len(snapshot), len(volume_list)))
    result_directory = os.path.join(result_path, result_name, 'volumes')
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    result_file = os.path.join(result_path, result_name, 'results.txt')
    output = open(result_file, 'w')
    output.close()

    for t in range(len(snapshot)):
        output = open(result_file, 'a+')
        output.write('Evaluating snapshot ' + str(epoch_list[t]) + ':\n')
        output.close()
        finished = True
        for i in range(len(volume_list)):
            volume_file = volume_filename_testing(result_directory, epoch_list[t], i)
            if not os.path.isfile(volume_file):
                finished = False
                break
        if not finished:
            net = RSTN(crop_margin=crop_margin, crop_prob=crop_prob, \
                       crop_sample_batch=crop_sample_batch, TEST='C').cuda()
            net.load_state_dict(torch.load(snapshot[t]))
            net.eval()

        for i in range(len(volume_list)):
            start_time = time.time()
            print('Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases, ' + \
                  str(t + 1) + ' out of ' + str(len(snapshot)) + ' snapshots.')
            volume_file = volume_filename_testing(result_directory, epoch_list[t], i)
            s = volume_list[i].split(' ')
            label = np.load(s[2])
            label = is_organ(label, organ_ID).astype(np.uint8)
            if not os.path.isfile(volume_file):
                image = np.load(s[1]).astype(np.float32)
                np.minimum(np.maximum(image, low_range, image), high_range, image)
                image -= low_range
                image /= (high_range - low_range)
                print('  Data loading is finished: ' + \
                      str(time.time() - start_time) + ' second(s) elapsed.')
                pred = np.zeros(image.shape, dtype=np.float32)
                minR = 0
                if plane == 'X':
                    maxR = image.shape[0]
                    shape_ = (1, 3, image.shape[1], image.shape[2])
                elif plane == 'Y':
                    maxR = image.shape[1]
                    shape_ = (1, 3, image.shape[0], image.shape[2])
                elif plane == 'Z':
                    maxR = image.shape[2]
                    shape_ = (1, 3, image.shape[0], image.shape[1])
                for j in range(minR, maxR):
                    if slice_thickness == 1:
                        sID = [j, j, j]
                    elif slice_thickness == 3:
                        sID = [max(minR, j - 1), j, min(maxR - 1, j + 1)]
                    if plane == 'X':
                        image_ = image[sID, :, :].astype(np.float32)
                    elif plane == 'Y':
                        image_ = image[:, sID, :].transpose(1, 0, 2).astype(np.float32)
                    elif plane == 'Z':
                        image_ = image[:, :, sID].transpose(2, 0, 1).astype(np.float32)

                    image_ = image_.reshape((1, 3, image_.shape[1], image_.shape[2]))
                    image_ = torch.from_numpy(image_).cuda().float()
                    out = net(image_).data.cpu().numpy()[0, :, :, :]

                    if slice_thickness == 1:
                        if plane == 'X':
                            pred[j, :, :] = out
                        elif plane == 'Y':
                            pred[:, j, :] = out
                        elif plane == 'Z':
                            pred[:, :, j] = out
                    elif slice_thickness == 3:
                        if plane == 'X':
                            if j == minR:
                                pred[j: j + 2, :, :] += out[1: 3, :, :]
                            elif j == maxR - 1:
                                pred[j - 1: j + 1, :, :] += out[0: 2, :, :]
                            else:
                                pred[j - 1: j + 2, :, :] += out[...]
                        elif plane == 'Y':
                            if j == minR:
                                pred[:, j: j + 2, :] += out[1: 3, :, :].transpose(1, 0, 2)
                            elif j == maxR - 1:
                                pred[:, j - 1: j + 1, :] += out[0: 2, :, :].transpose(1, 0, 2)
                            else:
                                pred[:, j - 1: j + 2, :] += out[...].transpose(1, 0, 2)
                        elif plane == 'Z':
                            if j == minR:
                                pred[:, :, j: j + 2] += out[1: 3, :, :].transpose(1, 2, 0)
                            elif j == maxR - 1:
                                pred[:, :, j - 1: j + 1] += out[0: 2, :, :].transpose(1, 2, 0)
                            else:
                                pred[:, :, j - 1: j + 2] += out[...].transpose(1, 2, 0)
                if slice_thickness == 3:
                    if plane == 'X':
                        pred[minR, :, :] /= 2
                        pred[minR + 1: maxR - 1, :, :] /= 3
                        pred[maxR - 1, :, :] /= 2
                    elif plane == 'Y':
                        pred[:, minR, :] /= 2
                        pred[:, minR + 1: maxR - 1, :] /= 3
                        pred[:, maxR - 1, :] /= 2
                    elif plane == 'Z':
                        pred[:, :, minR] /= 2
                        pred[:, :, minR + 1: maxR - 1] /= 3
                        pred[:, :, maxR - 1] /= 2
                print('  Testing is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.')
                pred = np.around(pred * 255).astype(np.uint8)
                np.savez_compressed(volume_file, volume=pred)
                print('  Data saving is finished: ' + \
                      str(time.time() - start_time) + ' second(s) elapsed.')
                pred_temp = (pred >= 128)
            else:
                volume_data = np.load(volume_file)
                pred = volume_data['volume'].astype(np.uint8)
                print('  Testing result is loaded: ' + \
                      str(time.time() - start_time) + ' second(s) elapsed.')
                pred_temp = (pred >= 128)

            DSC[t, i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_temp)
            print('    DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + \
                  ' + ' + str(label_sum) + ') = ' + str(DSC[t, i]) + ' .')
            output = open(result_file, 'a+')
            output.write('  Testcase ' + str(i + 1) + ': DSC = 2 * ' + str(inter_sum) + ' / (' + \
                         str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC[t, i]) + ' .\n')
            output.close()
            if pred_sum == 0 and label_sum == 0:
                DSC[t, i] = 0
            print('  DSC computation is finished: ' + \
                  str(time.time() - start_time) + ' second(s) elapsed.')

        print('Snapshot ' + str(epoch_list[t]) + ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .')
        output = open(result_file, 'a+')
        output.write('Snapshot ' + str(epoch_list[t]) + \
                     ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .\n')
        output.close()

    print('The testing process is finished.')
    for t in range(len(snapshot)):
        print('  Snapshot ' + str(epoch_list[t]) + ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .')




if __name__ == '__main__':

    coarse_testing_plane = 'A'


    if coarse_testing_plane == 'X' or coarse_testing_plane == 'A':
        coarse_testing(data_path,current_fold,organ_number,low_range,high_range,slice_threshold,slice_thickness,
                       organ_ID,'X',GPU_ID,learning_rate1,learinge_rate_m1,learning_rate2,learning_rate_m2,
                       margin,prob,sample_batch,training_epcoh_s,training_epcoh_i,training_epcon_j,lr_decay_epoch_j_step)

    if coarse_testing_plane == 'Y' or coarse_testing_plane == 'A':
        coarse_testing(data_path,current_fold,organ_number,low_range,high_range,slice_threshold,slice_thickness,
                       organ_ID,'Y',GPU_ID,learning_rate1,learinge_rate_m1,learning_rate2,learning_rate_m2,
                       margin,prob,sample_batch,training_epcoh_s,training_epcoh_i,training_epcon_j,lr_decay_epoch_j_step)


    if coarse_testing_plane == 'Z' or coarse_testing_plane == 'A':
        coarse_testing(data_path,current_fold,organ_number,low_range,high_range,slice_threshold,slice_thickness,
                       organ_ID,'Z',GPU_ID,learning_rate1,learinge_rate_m1,learning_rate2,learning_rate_m2,
                       margin,prob,sample_batch,training_epcoh_s,training_epcoh_i,training_epcon_j,lr_decay_epoch_j_step)

