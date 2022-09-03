import itertools
import os

# cmd
# string type:fixed parameter
# list type:varied parameter
run_cmd = 'python main.py'
model_prefix = '-ar'
models = ['vgg16_cifar100']
dataset_prefix = '-ds'
datasets = ['cifar100_raw']
method_prefix='-cr'
methods = ['lrp','grad','ICLR','weight']

# train= '--train false'
resume= '--resume true'
optimize_method= "-optm sgd"
# "save": false
epochs='--epochs 2'
lr = '--lr 0.001'
momentum = '-m 0.9'

prune = '--prune true'
prune_cnn = '--prune_cnn true'
prune_fc = '--prune_fc false'
prune_train = '--prune_train false'
# wait
classes = '--classes [12,82,86,89,94]'  # classes only use for cifar
samples = '--samples 5'

fine_tune_prefix = '--fine_tune'
fine_tunes = ['false',]
fine_tune_conv = '--fine_tune_conv true'
fine_tune_fc = '--fine_tune_fc true'

train_batch_size = '--train_batch_size 64'
test_batch_size = '--test_batch_size 64'
num_workers = '--num_workers 0'
pin_memory = '--pin_memory false'
persistent_workers= '--persistent_workers false'

# trialnum = '--trialnum false'
# weight_decay = '--weight_decay false'
seed_prefix = '--seed'
seeds = list(map(str,range(10)))
no_cuda = '--no_cuda false'
norm = '--norm true'
total_pr = '--total_pr 0.96'
pr_step = '--pr_step 0.05'
# showlrp= '--showlrp false'

log_name = '--log_name vgg16cifar100raw_C5_S5_NFT/log.txt'

save_dir = ''
# create bash file
save_bash_name = 'run.sh'


if save_dir:
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
save_path = os.path.join(save_dir,save_bash_name)

with open(save_path,'w') as bash_file:
    # os marker
    # bash_file.write('#!/bin/bash\n')
    i=0
    for model,dataset,method,fine_tune,seed in itertools.product(models,datasets,methods,fine_tunes,seeds):
        if i>0:
            bash_file.write('\n')
        cmd = ' '.join([run_cmd,
                        model_prefix,model,
                        dataset_prefix,dataset,
                        method_prefix,method,
                        resume,optimize_method,epochs,lr,momentum,
                        prune,prune_cnn,prune_fc,prune_train,
                        classes,samples,
                        fine_tune_prefix,fine_tune,fine_tune_conv,fine_tune_fc,
                        train_batch_size,test_batch_size,
                        num_workers,pin_memory,persistent_workers,
                        seed_prefix,seed,no_cuda,norm,total_pr,pr_step,
                        log_name])
        bash_file.write(cmd)
        i+=1







