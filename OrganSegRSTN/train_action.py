from utils import parameters
from OrganSegRSTN import training

if __name__ == '__main__':

    training_plane = 'A'


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

    if training_plane == 'X' or training_plane == 'A':
        training_model_name = 'x' + str(parameters.SLICE_THICKNESS) +'_'+ str(parameters.ORGAN_ID)
        training_log =  str(parameters.DATA_PATH) + '/logs/FD' + str(parameters.CURRENT_FLOD) + '_' + str(training_model_name) + '.txt'

        training.train(data_path,current_fold,organ_number,low_range,high_range,slice_threshold,slice_thickness,
                       organ_ID,'X',GPU_ID,learning_rate1,learinge_rate_m1,learning_rate2,learning_rate_m2,
                       margin,prob,sample_batch,training_epcoh_s,training_epcoh_i,training_epcon_j,
                       lr_decay_epoch_j_step,training_log)

    if training_plane == 'Y' or training_plane == 'A':
        training_model_name = 'Y' + str(parameters.SLICE_THICKNESS) +'_'+ str(parameters.ORGAN_ID)
        training_log =  str(parameters.DATA_PATH) + '/logs/FD' + str(parameters.CURRENT_FLOD) + '_' + str(training_model_name) + '.txt'

        training.train(data_path,current_fold,organ_number,low_range,high_range,slice_threshold,slice_thickness,
                       organ_ID,'Y',GPU_ID,learning_rate1,learinge_rate_m1,learning_rate2,learning_rate_m2,
                       margin,prob,sample_batch,training_epcoh_s,training_epcoh_i,training_epcon_j,
                       lr_decay_epoch_j_step,training_log)

    if training_plane == 'Z' or training_plane == 'A':
        training_model_name = 'Z' + str(parameters.SLICE_THICKNESS) +'_'+ str(parameters.ORGAN_ID)
        training_log =  str(parameters.DATA_PATH) + '/logs/FD' + str(parameters.CURRENT_FLOD) + '_' + str(training_model_name) + '.txt'

        training.train(data_path,current_fold,organ_number,low_range,high_range,slice_threshold,slice_thickness,
                       organ_ID,'Z',GPU_ID,learning_rate1,learinge_rate_m1,learning_rate2,learning_rate_m2,
                       margin,prob,sample_batch,training_epcoh_s,training_epcoh_i,training_epcon_j,
                       lr_decay_epoch_j_step,training_log)

