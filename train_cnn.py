from torch import nn

from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network_mcnn import MCNN
import torch.nn
import torch
from utils import log
import argparse
from config import get_config
import my_evaluation
import warnings

warnings.filterwarnings("ignore")

def train(config, task, suffix):
    train_set = GoTermDataset("train", task)
    pos_weights = torch.tensor(train_set.pos_weights).float()
    valid_set = GoTermDataset("val", task)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    output_dim = valid_set.y_true.shape[-1]
    model = MCNN(input_dim=512,hidden_dim=512, num_classes=output_dim,num_head=4).to(config.device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **config.optimizer,
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config.scheduler)
    bce_loss = torch.nn.BCELoss(reduce=False)

    train_loss = []
    val_loss = []
    val_aupr = []
    val_Fmax = []
    es = 0
    y_true_all = valid_set.y_true.float().reshape(-1)

    for ith_epoch in range(config.max_epochs):
        # scheduler.step()
        for idx_batch, batch in enumerate(train_loader):
            # with torch.autograd.set_detect_anomaly(True):
            model.train()
            # optimizer.zero_grad()
            y_pred = model(batch[0].to(config.device))
            # y_pred = y_pred.reshape([-1,2])
            y_true = batch[1].to(config.device)  # .reshape([-1])

            loss = bce_loss(y_pred, y_true)  # * pos_weights.to(config.device)
            loss = loss.mean()
            # loss = mlsm_loss(y_pred, y_true)

            log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss), 3)}")
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
                y_pred = model(batch[0].to(config.device))
                y_pred_all.append(y_pred)

            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)

            eval_loss = bce_loss(y_pred_all, y_true_all).mean()

            aupr = my_evaluation.macro_aupr(y_true_all.numpy(), y_pred_all.numpy())
            val_aupr.append(aupr)
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss), 3)} ||| aupr: {round(float(aupr), 3)}")
            val_loss.append(eval_loss.numpy())
            if ith_epoch == 0:
                best_eval_loss = eval_loss
                # best_eval_loss = aupr
            if eval_loss < best_eval_loss:
                # best_eval_loss = aupr
                best_eval_loss = eval_loss
                es = 0

                torch.save(model.state_dict(), config.cnn_model_save_path + f"{config.batch_size}_" + task + f"{suffix}1.pt")
            else:
                es += 1
                print("Counter {} of 5".format(es))

                # torch.save(model.state_dict(), config.model_save_path + task + f"{suffix}.pt")
            if es > 4:
                torch.save(
                    {
                        "train_bce": train_loss,
                        "val_bce": val_loss,
                        "val_aupr": val_aupr,
                    }, config.loss_save_path + f"cnn_{config.batch_size}_" + task + f"{suffix}1.pt"
                )

                break


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
    p.add_argument('--suffix', type=str, default='', help='')
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--batch_size', type=int, default=32, help='')

    args = p.parse_args()
    config = get_config()
    config.optimizer['lr'] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    if args.device != '':
        config.device = "cuda:" + args.device
    print(args)
    train(config, args.task, args.suffix)

