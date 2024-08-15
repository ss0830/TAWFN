from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network_agcn import AGCN
from network_mcnn import MCNN
from torch.nn.parameter import Parameter
import argparse
import pickle as pkl
from config import get_config
import my_evaluation
from utils import *
import torch.optim as optim
import torch.nn as nn


with open("data/ic_count.pkl", 'rb') as f:
    ic_count = pkl.load(f)
ic_count['bp'] = np.where(ic_count['bp'] == 0, 1, ic_count['bp'])
ic_count['mf'] = np.where(ic_count['mf'] == 0, 1, ic_count['mf'])
ic_count['cc'] = np.where(ic_count['cc'] == 0, 1, ic_count['cc'])
train_ic = {}
train_ic['bp'] = -np.log2(ic_count['bp'] / 69709)
train_ic['mf'] = -np.log2(ic_count['mf'] / 69709)
train_ic['cc'] = -np.log2(ic_count['cc'] / 69709)

class MyDatasets(Dataset):

    def __init__(self, n_view,y,x,x1):
        self.n_view = n_view
        self.y = y
        self.x = x
        self.x1 = x1

    def __getitem__(self, index):
        return self.x[index],self.x1[index],self.y[index]

    def __len__(self):
        return self.x.shape[0]
class SoftWeighted(nn.Module):
    def __init__(self, num_view):
        super(SoftWeighted, self).__init__()
        self.num_view = num_view
        self.weight_var = Parameter(torch.ones(num_view))

    def forward(self, data):
        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in range(self.num_view)]
        high_level_preds = 0
        for i in range(self.num_view):
            high_level_preds += weight_var[i] * data[i]

        return high_level_preds,weight_var

def test(config, task, model_gcn_pt, model_cnn_pt,test_type='test'):
    print(config.device)
    test_set = GoTermDataset(test_type, task)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    output_dim = test_set.y_true.shape[-1]
    weight1 = torch.tensor([0.5, 0.5])
    model_gcn = AGCN(output_dim, weight1, config.esmembed, config.pooling).to(config.device)
    model_cnn = MCNN(input_dim=512, hidden_dim=512, num_classes=output_dim, num_head=4).to(config.device)
    model_gcn.load_state_dict(torch.load(model_gcn_pt, map_location=config.device))
    model_cnn.load_state_dict(torch.load(model_cnn_pt, map_location=config.device))
    model_gcn.eval()
    model_cnn.eval()
    termIC = train_ic[task]
    bce_loss = torch.nn.BCELoss().to(config.device)

    EPOCHS = 35
    y_pred_all = []
    y_pred_all_gcn = []
    y_pred_all_cnn = []

    y_true_all = test_set.y_true.float()
    weight_var_list = []
    with torch.no_grad():
        for idx_batch, batch in enumerate(test_loader):
            # model.eval()
            # y_pred, _, _ = model(batch[0].to(config.device))
            y_pred_gcn = model_gcn(batch[0].to(config.device))
            y_pred_cnn = model_cnn(batch[0].to(config.device))

            y_pred_all_gcn.append(y_pred_gcn)
            y_pred_all_cnn.append(y_pred_cnn)

        y_pred_all_gcn = torch.cat(y_pred_all_gcn, dim=0).cpu()
        y_pred_all_cnn = torch.cat(y_pred_all_cnn, dim=0).cpu()
        y_pred_all.append(y_pred_all_gcn)
        y_pred_all.append(y_pred_all_cnn)

    deal_dataset = MyDatasets(2, y_true_all, y_pred_all_gcn, y_pred_all_cnn)
    awf_loader = DataLoader(deal_dataset, batch_size=config.batch_size, shuffle=False)
    net = SoftWeighted(2).to(config.device)
    optimizer = optim.Adam(net.parameters(), lr=5e-3)
    lr_sched = True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_alpha = None
    best_epoch = 0
    f_max = -1
    for epoch in range(EPOCHS):
        test_loss = 0.0
        net.train()
        for batch_idx, data in enumerate(awf_loader):
            # print(f"Batch Index: {batch_idx}")
            data = [x.to(config.device) for x in data]
            labels = data.pop()
            output, weight_var = net(data)
            loss = bce_loss(output, labels)
            test_loss += loss.item()
            loss.backward()
            optimizer.step()

        weight_var = None
        with torch.no_grad():  # set all 'requires_grad' to False
            for data in awf_loader:
                data = [x.to(config.device) for x in data]
                labels = data.pop()
                output, weight_var = net(data)
                break
        weight_var_list.append(weight_var)
        test_loss /= len(awf_loader)
        weight_var = [weight_var[i].item() for i in range(2)]
        fusion_pre = torch.zeros(y_pred_all[0].shape)
        for i in range(2):
            fusion_pre += weight_var[i] * y_pred_all[i]

        avg_fmax = my_evaluation.fmax(y_true_all.numpy(), fusion_pre.numpy(), task)
        avg_loss = bce_loss(fusion_pre,y_true_all)

        print(f"--- weight val epoch_{epoch} : {weight_var}")
        print("--- test Loss:            %.4f" % avg_loss)
        print("--- test F-score:         %.4f" % avg_fmax)

        if avg_fmax > f_max:
            best_epoch = epoch
            best_alpha = weight_var
            f_max = avg_fmax
            es = 0
        else:
            es += 1
            print("Counter {} of 5".format(es))
        if es > 4:
            break

        if lr_sched:
            scheduler.step(test_loss)

    print(f"[*] Loaded checkpoint at best epoch {best_epoch}:")

    fusion_pre = torch.zeros(y_pred_all[0].shape)
    for i in range(2):
        fusion_pre += best_alpha[i] * y_pred_all[i]

    eval_loss = bce_loss(fusion_pre, y_true_all)
    # Fmax = fmax(y_true_all.numpy(), y_pred_all.numpy())
    Fmax = my_evaluation.fmax(y_true_all.numpy(), fusion_pre.numpy(), task)
    # aupr = metrics.average_precision_score(y_true_all.numpy(), y_pred_all.numpy(), average='samples')
    aupr = my_evaluation.macro_aupr_test(y_true_all.numpy(), fusion_pre.numpy())
    Smin = my_evaluation.smin(termIC, y_true_all.numpy(), fusion_pre.numpy())
    log(f"Test ||| loss: {round(float(eval_loss), 3)} ||| aupr: {round(float(aupr), 3)} ||| Fmax: {round(float(Fmax), 3)}||| Smin: {round(float(Smin), 3)}")
    torch.save(
        {
            "weight_var": weight_var_list,
            "loss": eval_loss,
            "aupr" : aupr,
            "Fmax" : Fmax,
            "Smin" :Smin
        }, config.test_result_path + task + f".pt"
    )
    # if test_type == 'AF2test':
    #     result_name = config.test_result_path + 'AF2'+ model_pt[6:]
    # else:
    # result_name = config.test_result_path + model_pt[5:]
    # with open(result_name, "wb") as f:
    #     pkl.dump([y_pred_all.numpy(), y_true_all.numpy()], f)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='bp', choices=['bp', 'mf', 'cc'], help='')
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--model_gcn', type=str, default='', help='')
    p.add_argument('--model_cnn', type=str, default='', help='')
    p.add_argument('--esmembed', default=True, type=str2bool, help='')
    p.add_argument('--pooling', default='MHA', type=str, choices=['MHA', 'MP'],
                   help='Multi-head attention or Global mean pooling')
    p.add_argument('--AF2test', default=False, type=str2bool, help='')

    args = p.parse_args()
    print(args)
    config = get_config()
    config.batch_size = 32
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.device != '':
        config.device = "cuda:" + args.device
    config.esmembed = args.esmembed
    config.pooling = args.pooling
    if not args.AF2test:
        test(config, args.task, args.model_gcn, args.model_cnn)
    else:
        test(config, args.task, args.model_gcn, args.model_cnn, 'AF2test')
