import dill
import torch.nn.functional as F
from tqdm.auto import tqdm
from test_kagsl import *


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def train(model=None,
          save_path='',
          config={},
          train_dataloader=None,
          val_dataloader=None,
          ):

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['decay'])

    train_loss_list = []
    val_loss_list = []

    epoch = config['epoch']

    min_loss = 1e+8
    early_stop_win = config['early_stop_win']
    stop_improve_count = 0

    for i_epoch in range(epoch):
        i = 0
        acu_loss = 0

        model.train()
        with tqdm(train_dataloader) as it:
            for x, y, attack_labels in it:

                x = x.float()
                y = y.float()

                optimizer.zero_grad()
                out = model(x)
                loss = loss_func(out, y)

                loss.backward()
                optimizer.step()

                i += 1

                acu_loss += loss.item()
                avg_acu_loss = acu_loss / i

                it.set_postfix(
                    {
                        "epoch": f"{i_epoch+1}/{epoch}",
                        "avg_loss": avg_acu_loss,
                    },
                    refresh=False,
                )
            it.close()

        train_loss_list.append(avg_acu_loss)

        if val_dataloader is not None:
            avg_val_loss, _ = test(model, val_dataloader)
            val_loss_list.append(avg_val_loss)

            if avg_val_loss < min_loss:
                torch.save(model, save_path, pickle_module=dill)
                print("validation loss {} -> {}".format(min_loss, avg_val_loss))

                min_loss = avg_val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1
                print("current val loss {} vs current min val loss {}".format(avg_val_loss, min_loss))
                print("early stopping patience count {} / {}".format(stop_improve_count, early_stop_win))

        else:
            if avg_acu_loss < min_loss:
                torch.save(model, save_path, pickle_module=dill)
                print("train loss {} -> {}".format(min_loss, avg_acu_loss))

                min_loss = avg_acu_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1
                print("current train loss {} vs current min train loss {}".format(avg_acu_loss, min_loss))
                print("early stopping patience count {} / {}".format(stop_improve_count, early_stop_win))

        if stop_improve_count > early_stop_win:
            break

    return train_loss_list, val_loss_list
