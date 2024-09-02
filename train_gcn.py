from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network_agcn import AGCN
from contrastive_loss import ContrastiveLoss
import torch.nn
import torch
from utils import log
import argparse
from config import get_config
import my_evaluation
import warnings
warnings.filterwarnings("ignore")
def train(config, task):

    train_set = GoTermDataset("train", task)
    valid_set = GoTermDataset("val", task)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    # net = SoftWeighted(n_view).to(config.device)
    output_dim = valid_set.y_true.shape[-1]
    model = AGCN(output_dim, 0, config.esmembed, config.pooling, config.contrast).to(config.device)
    optimizer = torch.optim.Adam(
        params = model.parameters(), 
        **config.optimizer,
        )

    bce_loss = torch.nn.BCELoss(reduce=False)

    train_loss = []
    val_loss = []
    val_aupr = []
    val_Fmax = []
    es = 0
    y_true_all = valid_set.y_true.float().reshape(-1)

    for ith_epoch in range(config.max_epochs):
        for idx_batch, batch in enumerate(train_loader):
            model.train()
            
            if config.contrast:
                y_pred,  g_feat1, g_feat2, _= model(batch[0].to(config.device))
                # y_pred, _ = net([y_pred1,y_pred2])
                y_true = batch[1].to(config.device)
                _loss = bce_loss(y_pred, y_true)
                _loss = _loss.mean()
                criterion = ContrastiveLoss(g_feat1.shape[0], 0.1, 1)
                cl_loss = 0.05 * criterion(g_feat1, g_feat2)
                # cl_loss_cnn = 0.05 * criterion_cnn(cnn_1, cnn_2)

                loss = _loss + cl_loss
            else:
                y_pred= model(batch[0].to(config.device))


                y_true = batch[1].to(config.device)

                loss = bce_loss(y_pred, y_true)
                loss = loss.mean()
                # loss = mlsm_loss(y_pred, y_true)

            log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)}")
            train_loss.append(loss.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        eval_loss = 0
        model.eval()
        y_pred_all = []
        n_nce_all = []
        
        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                if config.contrast:
                    y_pred,_,_,wg1 = model(batch[0].to(config.device))

                else:
                    y_pred = model(batch[0].to(config.device))

                y_pred_all.append(y_pred)
            
            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)
            
            eval_loss = bce_loss(y_pred_all, y_true_all).mean()
                
            aupr = my_evaluation.macro_aupr(y_true_all.numpy(), y_pred_all.numpy())
            val_aupr.append(aupr)
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)} ||| aupr: {round(float(aupr),3)}")
            val_loss.append(eval_loss.numpy())
            if ith_epoch == 0:
                best_eval_loss = eval_loss
                #best_eval_loss = aupr
            if eval_loss < best_eval_loss:
                #best_eval_loss = aupr
                best_eval_loss = eval_loss
                es = 0

                torch.save(model.state_dict(), config.gcn_model_save_path + f"_" + task + f".pt")
                if config.contrast:
                    print(wg1)
            else:
                es += 1
                print("Counter {} of 5".format(es))

            if es > 4:
                
                torch.save(
                    {
                        "train_bce": train_loss,
                        "val_bce": val_loss,
                        "val_aupr": val_aupr,
                    }, config.loss_save_path + f"gcn_{config.batch_size}_" + task + f".pt"
                )

                break

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='bp', choices=['bp','mf','cc'], help='')
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--esmembed', default=True, type=str2bool, help='')
    p.add_argument('--pooling', default='MHA', type=str, choices=['MHA','MP'], help='Multi-head attention or Global mean pooling')
    p.add_argument('--contrast', default=False, type=str2bool, help='whether to do contrastive learning')
    p.add_argument('--batch_size', type=int, default=32, help='')

    args = p.parse_args()
    config = get_config()
    config.optimizer['lr'] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    if args.device != '':
        config.device = "cuda:" + args.device
    config.esmembed = args.esmembed
    print(args)
    config.pooling = args.pooling
    config.contrast = args.contrast
    train(config, args.task)

