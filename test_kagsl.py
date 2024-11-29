import torch
import torch.nn as nn

from tqdm import tqdm


def test(model, dataloader):
    loss_func = nn.MSELoss(reduction='none')  # mean -> sum

    predict_list = []
    ground_list = []
    label_list = []

    acu_loss = 0

    model.eval()
    with tqdm(dataloader) as it_test:
        for x, y, labels in it_test:
            x = x.float()
            y = y.float()

            with torch.no_grad():
                out = model(x)
                loss = loss_func(out, y).mean(-1).sum(-1)  #
                acu_loss += loss.item()

            predict_list.append(out)
            ground_list.append(y)
            label_list.append(labels)

        it_test.close()

    predict_pt = torch.cat(predict_list, dim=0)
    ground_pt = torch.cat(ground_list, dim=0)
    label_pt = torch.cat(label_list, dim=0)

    avg_acu_loss = acu_loss / predict_pt.shape[0]

    return avg_acu_loss, (predict_pt.cpu(), ground_pt.cpu(), label_pt.cpu())
