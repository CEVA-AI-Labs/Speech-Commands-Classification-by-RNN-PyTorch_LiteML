if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import torch 
from clearml import Task

from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type

if 1: # my lib
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons
    import utils.lib_datasets as lib_datasets
    import utils.lib_augment as lib_augment
    import utils.lib_ml as lib_ml
    import utils.lib_rnn as lib_rnn

import sys
sys.path.append("/home/olyas/ailabs_gru/ailabs_qat")
sys.path.append("/home/olyas/ailabs_gru/ailabs_shared")
sys.path.append("/home/olyas/ailabs_gru/ailabs_pruning")
sys.path.append("/home/olyas/ailabs_gru/ailabs_liteml")
from torch.utils.tensorboard import SummaryWriter
from  liteml.retrainer import RetrainerModel,RetrainerConfig


class CustomTracer(torch.fx.Tracer):

    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        fn, args = super().create_args_for_root(root_fn, is_module, concrete_args)
        args[1].size = lambda x: 1
        return fn, args



cfg = RetrainerConfig("/home/olyas/ailabs_lstm/ailabs_tests/classification/configs/test_config.yaml", custom_tracer= CustomTracer)
data_bits = cfg.optimizations_config['QAT']['data-quantization']['bits']
w_bits = cfg.optimizations_config['QAT']['weights-quantization']['bits']
task = Task.init(project_name='Regression tests/lstm' , task_name="lstm retraining", tags=["{}/{}".format(data_bits, w_bits)])
if not os.path.exists("tensorboard"):
    os.makedirs("tensorboard")
writer = SummaryWriter(log_dir= "tensorboard")


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------


# Set arguments ------------------------- 
args = lib_rnn.set_default_args()

args.learning_rate = 0.001
args.num_epochs = 50
args.learning_rate_decay_interval = 10 # decay for every 3 epochs
args.learning_rate_decay_rate = 0.5 # lr = lr * rate
args.do_data_augment = True
args.train_eval_test_ratio=[0.8, 0.195, 0.005]
args.data_folder = "data/kaggle/"
args.classes_txt = "config/classes_kaggle.names" 
args.load_weights_from ="checkpoints_3layerbidirectional/002.ckpt"

# Dataset -------------------------- 

# Get data's filenames and labels
files_name, files_label = lib_datasets.AudioDataset.load_filenames_and_labels(
    args.data_folder, args.classes_txt)

# if 1: # DEBUG: use only a subset of all data
#     GAP = 1000
#     files_name = files_name[::GAP]
#     files_label = files_label[::GAP]
#     args.num_epochs = 5
    
# Set data augmentation
if args.do_data_augment:
    Aug = lib_augment.Augmenter # rename
    aug = Aug([        
        Aug.Shift(rate=0.2, keep_size=False), 
        Aug.PadZeros(time=(0, 0.3)),
        Aug.Amplify(rate=(0.2, 1.5)),
        # Aug.PlaySpeed(rate=(0.7, 1.3), keep_size=False),
        Aug.Noise(noise_folder="data/noises/", 
                        prob_noise=0.7, intensity=(0, 0.7)),
    ], prob_to_aug=0.8)
else:
    aug = None

# Split data into train/eval/test
tr_X, tr_Y, ev_X, ev_Y, calib_X, calib_Y = lib_ml.split_train_eval_test(
    X=files_name, Y=files_label, ratios=args.train_eval_test_ratio, dtype='list')
train_dataset = lib_datasets.AudioDataset(files_name=tr_X, files_label=tr_Y, transform=aug)
eval_dataset = lib_datasets.AudioDataset(files_name=ev_X, files_label=ev_Y, transform=None)
calibration_dataset = lib_datasets.AudioDataset(files_name=calib_X, files_label=calib_Y, transform=None)



# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False)
calibration_loader = torch.utils.data.DataLoader(dataset=calibration_dataset, batch_size=args.batch_size, shuffle=False)


sample, classes = next(iter(calibration_loader))
sample = sample.cuda()
# Create model and train -------------------------------------------------
model = lib_rnn.create_RNN_model(args, load_weights_from="checkpoints_RNN/003.ckpt", bidirectional=False)# #

out1 = model(sample)
print("Accuracy before quantization : ")
lib_rnn.evaluate_model(model, calibration_loader, num_to_eval=-1, writer=writer, epoch=-2)


wrap_in_retrainer = True
if wrap_in_retrainer:
    rmodel = RetrainerModel(model, cfg)
    #out2 =rmodel(sample)
    #print("")
    rmodel.initialize_quantizers(calibration_loader, lambda model, x:model(x[0].cuda()))
    print("Accuracy before quantization after wrapping : ")
    lib_rnn.evaluate_model(rmodel, calibration_loader, num_to_eval=-1, writer=writer, epoch=-2)
    out2 =rmodel(sample)
else:
    rmodel =model


do_qat = True

if do_qat:
    rmodel = torch.nn.DataParallel(rmodel, [0,1,2,3,4,5,6,7]).cuda()
    rmodel = rmodel.cuda()
    lib_rnn.train_model(rmodel, args, train_loader, eval_loader, writer)


# GRU 3 layer : Accuracy =  83.58490566037736%
#8 bit quantization:  76.22641509433963%
#after retraining:
