"""
Utilities related to model evaluation
==> Inspired by the GitHub repository of Wood et al. 2022 (https://github.com/MIDIconsortium/BrainAge)
"""

import torch


def evaluate(net, data_loader, criterion, device, eval_type, print_message=False):
    """ Validation or test function

    :param net: torch neural network
    :param data_loader: torch DataLoader
    :param criterion: torch.nn loss criterion. CAVE: use "sum" as reduction!
    :param device: torch.device, cpu or gpu
    :param eval_type: str, choose from ["validation", "test"]
    :param print_message: bool, print message?
    :return: validation/test loss, true label list, pred label list
    """
    # Initializations and set device
    loss_sum = 0
    samples_count = 0
    true_label_list = []
    pred_label_list = []

    with torch.no_grad():
        net.eval().to(device)
        for img, label in data_loader:
            # Send to device
            img = img.to(device)
            label = label.to(device)

            # Predict label
            pred_label = net(img)
            pred_label = torch.squeeze(pred_label, dim=1)  # Remove one dimension to match dimensionality of label

            # Extend lists and sum variables to allow calculation of mean loss
            # pred_label_list.extend((np.e**pred_label).argmax(dim=1).tolist())
            pred_label_list.extend(pred_label.reshape(1, -1)[0].tolist())
            true_label_list.extend(label.reshape(1, -1)[0].tolist())
            loss_sum += criterion(pred_label, label).detach().item()
            samples_count += img.shape[0]

        loss_mean = loss_sum / samples_count
        if print_message:
            print(f'{eval_type}: loss: {loss_mean}')

        return loss_mean, true_label_list, pred_label_list
