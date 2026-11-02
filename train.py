import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import  get_evaluation_results
from Dataloader import load_data
from Dataloader_IsoGraph import load_Iso_data
from model import CrossFormer
import scipy.sparse as ss
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import triton
# import triton.language as tl

def train(args, device):
    # if dataset_name in ["YTF200",'YTF100',"YTF50"]:
    #     path = os.path.join(args.path, dataset_name)
    #     mat = hdf5storage.loadmat(path)
    #     features = mat['X']
    #     gnd = np.squeeze(mat['Y']) - np.min(mat['Y'])
    feature_list, labels, idx_labeled, idx_unlabeled = load_data(args, device)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    N = feature_list[0].shape[0]
    num_view = len(feature_list)
    # print(N,num_view,num_classes)
    input_dims = []
    for i in range(num_view): # multiview data { data includes features and ... }
        input_dims.append(feature_list[i].shape[1])

    model = CrossFormer(input_dims, args.hidden_channels, num_classes, num_view, args).to(device)
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Best_Acc = 0.
    Loss_list = []
    ACC_list = []
    F1_list = []
    begin_time = time.time()
    with tqdm(total=args.num_epoch, desc="training", position=0) as pbar:
        for epoch in range(args.num_epoch):
            #encoder
            model.train()
            output, loss0 = model(feature_list)
            output = F.log_softmax(output, dim=1)
            optimizer.zero_grad()
            loss1 = loss_function(output[idx_labeled], labels[idx_labeled])
            loss = args.lamda * loss0 + loss1
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                output,_ = model(feature_list)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, _, _, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
                pbar.set_postfix({'Loss_ce': '{:.6f}'.format(loss.item()), 'ACC': '{:.2f}'.format(ACC * 100)})
                pbar.update(1)
                Loss_list.append(float(loss.item()))
                ACC_list.append(ACC)
                F1_list.append(F1)

    # draw_tsne(args.dataset, output, labels)
    cost_time = time.time() - begin_time
    model.eval()
    output,_ = model(feature_list)


    print("Evaluating the model")
    pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
    ACC, P, R, F1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled])
    print("------------------------")
    print("ratio = ",args.ratio)
    print("ACC:   {:.2f}".format(ACC * 100))
    print("F1 :   {:.2f}".format(F1 * 100))
    print("------------------------")

    return ACC, P, R, F1, cost_time, Loss_list, ACC_list, F1_list














def score(prediction, labels):
    prediction = prediction.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def evaluateT(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1