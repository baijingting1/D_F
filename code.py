import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from tqdm import tqdm
#from model7 import InceptionResNetV2
# from Inception_ResNet import InceptionResNetV2
#from demo import InceptionResNetV2
from finally_f import InceptionResNetV2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib import rcParams
directory = '/D:/研/DC-PC-Dilated-IR-V2/core_code_230325'

def main():
    '''-----------------------------数据处理--------------------------------------------'''
    data_transform = {
        "train": transforms.Compose([
                                     transforms.Resize((224,224)),
                                     transforms.Grayscale(num_output_channels=3),
                                     transforms.RandomHorizontalFlip(p=0.8),
                                     transforms.RandomRotation(degrees = 30, expand=False),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5], std=[0.5])]),
        "val": transforms.Compose([transforms.Resize((224,224)),
                                   transforms.Grayscale(num_output_channels=3),
                                   transforms.CenterCrop(224),
                                   transforms.RandomHorizontalFlip(p=0.8),
                                   transforms.RandomRotation(degrees=30, expand=False),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5], std=[0.5])])}


    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "/mnt/core_code","dataset1")
    #image_path = os.path.join(data_root, "core_code_230325","dataset1")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    print(flower_list)
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    #DataLoader作为一个迭代器，每次会产生一个batch size大小的数据。
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #nw=8
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size , shuffle=True,
                                               num_workers = nw,drop_last=True)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size= batch_size , shuffle=False,
                                                  num_workers=nw,drop_last=True)
    print("using {} images for training, {} images for validation.".format(train_num,val_num))
    '''-----------------------------引入网络模型、损失函数、优化器--------------------------------------------'''
    net = InceptionResNetV2()
   # device_ids=(1,2,3,4,5,6)
   # device = torch.nn.DataParallel(net, device_ids=device_ids)
   # device = torch.nn.DataParallel(net, device_ids=[0])
    device = torch.device("cuda:0") #定义训练的设备
   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)  #
    loss_function = nn.CrossEntropyLoss() #定义损失函数，这里使用交叉熵损失函数
    params = [p for p in net.parameters() if p.requires_grad] #将网络中所有需要更新梯度的参数放入params
    optimizer = optim.Adam(params, lr=0.0002)# 构建优化器，传入两个参数，前者为要优化的参数，后者为学习率
    epochs = 130
    Loss_plot = {}
    train_prec1_plot = {}
    val_prec1_plot = {}
    train_prec1=[]
    Loss=[]
    Val_Loss=[]
    val_prec1=[]
    for epoch in range(epochs):
        net.train() #net.train()标识了当前模型处于训练还是测试阶段。在模型训练时，前面必须加上net.train()。同样模型验证和测试时必须加上net.eval()。这个主要是为了更好的处理 Batch Normalization 和 Dropout。
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        train_acc = 0.0
        for step, data in enumerate(train_bar):
            torch.cuda.synchronize()
           # images, labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
            images, labels = data
            optimizer.zero_grad() #就是把梯度置零，也就是把loss关于weight的导数变成0。在训练过程中，每一个batch会更新一次网络的梯度，当到下一个batch计算梯度时，前一个batch的梯度已经没用了，所以需要将其变成0。
            logits = net(images.to(device)) #就是将图片张量送入网络前向传播，images.to(device) 是让其在GPU上计算，注意，前面的net.to(device) 是让网络的参数在GPU上计算，而Pytorch不允许不同设备的参数一起计算，所以也需要将图片张量设置为GPU设备计算。
            loss = loss_function(logits, labels.to(device)) # 计算损失，这里的loss是一个batch的平均损失
           # predict_y = torch.max(logits, dim=1)[1]
            predict_y = torch.max(logits, 1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            #Loss_plot[epoch] = loss.item()
            loss.backward() #反向传播，根据loss计算梯度。
            optimizer.step() #更新梯度
            running_loss += loss.item() #训练进度条
           # torch.cuda.synchronize()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch,epochs,loss)

        '''-----------------------------验证过程--------------------------------------------'''
        net.eval()
        val_acc = 0.0
        valing_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            y_true_val_total = None
            y_predict_val_total = None
            for val_data in val_bar:
               #val_images, val_labels = val_images.cuda(device=device_ids[0]), val_labels.cuda(device=device_ids[0])
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                optimizer.zero_grad()  # 就是把梯度置零，也就是把loss关于weight的导数变成0。在训练过程中，每一个batch会更新一次网络的梯度，当到下一个batch计算梯度时，前一个batch的梯度已经没用了，所以需要将其变成0。
                logits = net(val_images.to(device))  # 就是将图片张量送入网络前向传播，images.to(device) 是让其在GPU上计算，注意，前面的net.to(device) 是让网络的参数在GPU上计算，而Pytorch不允许不同设备的参数一起计算，所以也需要将图片张量设置为GPU设备计算。
                val_loss = loss_function(logits, val_labels.to(device))
            #predict_y = torch.max(outputs, dim=1)[1]
                predict_y = torch.max(outputs, 1)[1]
                if y_true_val_total == None:
                    y_true_val_total = val_labels.cpu().data
                    y_predict_val_total = predict_y.cpu()
                y_true_val_total = torch.cat([y_true_val_total, val_labels.cpu().data], 0)
                y_predict_val_total = torch.cat([y_predict_val_total, predict_y.cpu()], 0)
               # y_true_val_total = torch.cat([y_true_val_total, val_labels.cpu().data],dim=0)
               # y_predict_val_total = torch.cat([y_predict_val_total, predict_y.cpu()],dim=0)
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                valing_loss += val_loss.item()  # 训练进度条
                val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}".format(epoch,epochs,val_loss)
        train_accurate = train_acc / train_num
        val_accurate = val_acc / val_num
        train_prec1.append(100 * train_acc / (len(train_dataset)))
        Loss.append(loss)
        Val_Loss.append(val_loss)

        val_prec1.append(100 * val_acc / (len(validate_dataset)))
        train_prec1_plot[epoch] = train_accurate
        val_prec1_plot[epoch] = val_accurate

    print("train_accurate",train_prec1_plot)
    print("test_accurate",val_prec1_plot)

    #print("loss",Loss)
    #
    # config = {
    #     "font.family": 'serif',
    #      # 相当于小四大小
    #     "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    #     "font.serif": ['Times New Roman'],  # 宋体
    #     'axes.unicode_minus': False  # 处理负号，即-号
    # }
    # rcParams.update(config)

    x1 = range(0, 130)
    x2 = range(0, 130)
    y1 = train_prec1
    y3 = val_prec1
    y2 = Loss
    y4 = Val_Loss
    plt.figure(1)
    plt.subplot(2, 1, 2)
    plt.plot(x1, y1,label='train')
    plt.plot(x1,y3,label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy/%')
    plt.legend()
    plt.figure(2)
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2,label='train' )
    plt.plot(x2, y4,label='test')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
