import torch.utils.data

from penn_fudan_dataset import *
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from yolo_model_base import *
import cv2

datapath = '../dataset/PennFudanPed'
dataset = PennFudanDataset(datapath, get_transform(train=False))
dataset_test = PennFudanDataset(datapath, get_transform(train=False))

# indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[0: 6])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[0: 2])


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)


def input_process(batch):
    batch_size = len(batch[0])
    input_batch = torch.zeros(batch_size, 3, 448, 448)
    for i in range(batch_size):
        inputs_tmp = Variable(batch[0][i])
        inputs_tmp1 = cv2.resize(inputs_tmp.permute([1, 2, 0]).numpy(), (448, 448))
        inputs_tmp2 = torch.tensor(inputs_tmp1).permute([2, 0, 1])
        input_batch[i: i+1, :, :, :] = torch.unsqueeze(inputs_tmp2, 0)
    return input_batch


def target_process(batch):
    batch_size = len(batch[0])
    target_batch = torch.zeros(batch_size, 1, 1, 5)

    for i in range(batch_size):
        bbox = batch[1][i]['boxes'][0]
        _, hi, wi = batch[0][i].numpy().shape
        bbox = bbox / torch.tensor([wi, hi, wi, hi])
        cbbox = torch.cat([torch.ones(1), bbox])
        target_batch[i: i+1, :, :, :] = torch.unsqueeze(cbbox, 0)
    return target_batch


num_classes = 2
n_class = 2
batch_size = 6
epochs = 1
lr = 1e-3
momentum = 0
w_decay = 1e-5
step_size = 50
gamma = 0.5

yolo_model = YOLO()
optimizer = optim.SGD(yolo_model.detector.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
# decay LR by a factor of 0.5 every 50 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def loss_func(outputs, labels):
    tmp = (outputs - labels) ** 2
    return torch.sum(tmp, 0).view(1, 5).mm(torch.tensor([10, 0.0001, 0.0001, 0.0001, 0.0001]).view(5, 1))


def loss_func_details(outputs, labels):
    b, w, h, c = outputs.shape
    loss = 0
    for bi in range(b):
        for wi in range(w):
            for hi in range(h):
                detect_vector = outputs[bi, wi, hi]
                gt_dv = labels[bi, wi, hi]
                conf_pred = detect_vector[0]
                conf_gt = gt_dv[0]
                x_pred = detect_vector[1]
                x_gt = gt_dv[1]
                y_pred = detect_vector[2]
                y_gt = gt_dv[2]
                w_pred = detect_vector[3]
                w_gt = gt_dv[3]
                h_pred = detect_vector[4]
                h_gt = gt_dv[4]
                loss_confidence = (conf_pred - conf_gt) ** 2
                loss_geo = (x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2 + (w_pred - w_gt) ** 2 + (h_pred - h_gt) ** 2
                loss_tmp = loss_confidence + 0.3 * loss_geo
                loss += loss_tmp
    return loss


def train():
    for epoch in range(epochs):
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = input_process(batch)
            labels = target_process(batch)
            outputs = yolo_model(inputs)
            loss = loss_func_details(outputs, labels)
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("epoch:{}, iter:{}, loss:{}, lr:{}".format(epoch, iter, loss.data.item(),
                                                                 optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler.step()


def val():
    yolo_model.eval()
    for iter, batch in enumerate(val_loader):
        inputs = input_process(batch)
        target = target_process(batch)

        output = yolo_model(inputs)
        output = output.data.cpu().numpy()
        N, h, w, _ = output.shape
        pred = output.reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)


if __name__ == "__main__":
    train()
    val()

