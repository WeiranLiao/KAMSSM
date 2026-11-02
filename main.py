import os
import warnings
import random
import torch
import configparser
import numpy as np
from args import parameter_parser
from utils import tab_printer
from train import train
from scipy.io import savemat
from plot_function import draw_tsne
import time
torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    torch.autograd.set_detect_anomaly(True)


    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)

    all_ACC = []
    all_F1 = []
    all_TIME = []





    for i in range(args.n_repeated):
        ACC, P, R, F1, cost_time, Loss_list, ACC_list, F1_list = train(args, device)
        all_ACC.append(ACC)
        all_F1.append(F1)
        all_TIME.append(cost_time)
    print("====================")
    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
    print("====================")
    if args.save_results:
        experiment_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        results_direction = './results/' + args.dataset + '_results.txt'
        fp = open(results_direction, "a+", encoding="utf-8")
        fp.write(format(experiment_time))
        fp.write("\ndataset_name: {}\n".format(args.dataset))
        fp.write("alpha: {}  |  ".format(args.alpha))
        fp.write("layers: {}  |  ".format(args.layers))
        fp.write("hidden_channels: {}  |  ".format(args.hidden_channels))
        fp.write("ratio: {}  |  ".format(args.ratio))
        fp.write("epochs: {}  |  ".format(args.num_epoch))
        fp.write("lr: {}  |  ".format(args.lr))
        fp.write("wd: {}\n".format(args.weight_decay))
        fp.write("temp: {}  |  ".format(args.temp))
        fp.write("eta: {}  |  ".format(args.eta))
        fp.write("lamda: {}  |  ".format(args.lamda))
        fp.write("delta: {}\n".format(args.delta))
        fp.write("ACC:  {:.2f} ({:.2f})\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
        fp.write("F1 :  {:.2f} ({:.2f})\n".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
        fp.write("Time:  {:.2f} ({:.2f})\n\n".format(np.mean(all_TIME), np.std(all_TIME)))
        fp.close()

    # if args.save_all:
    #     if args.save_loss:
    #         fp2 = open("results/loss/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp2.seek(0)
    #         fp2.truncate()
    #         for i in range(len(Loss_list)):
    #             fp2.write(str(Loss_list[i]) + '\n')
    #         fp2.close()
    #
    #     if args.save_ACC:
    #         fp3 = open("results/ACC/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp3.seek(0)
    #         fp3.truncate()
    #         for i in range(len(ACC_list)):
    #             fp3.write(str(ACC_list[i]) + '\n')
    #         fp3.close()
    #
    #     if args.save_F1:
    #         fp4 = open("results/F1/" + str(args.dataset) + ".txt", "a+", encoding="utf-8")
    #         fp4.seek(0)
    #         fp4.truncate()
    #         for i in range(len(F1_list)):
    #             fp4.write(str(F1_list[i]) + '\n')
    #         fp4.close()