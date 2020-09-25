import os
import random
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from model1 import AttnVGG_before
from model2 import *
from utilities import *
from dataset import *
from Optimizer_torch import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR100")

parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')

parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
#parser.add_argument("--no_attention", action='store_true', help='turn down attention')
parser.add_argument("--log_images", action='store_true', help='log images and (is available) attention maps')

opt = parser.parse_args()


def main(train=True):
    # load data
    # CIFAR-100: 500 training images and 100 testing images per class
    print('\nloading the dataset ...\n')

    trainset = IMG_Dataset(train=True,  dataset_path='/media/wenyan/data_nataliya/avant/data_processing/data/im_attention_dataset/')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=5)
    #testset = IMG_Dataset(train=False,  dataset_path='/media/wenyan/data_nataliya/avant/data_processing/data/im_attention_dataset/')
    #testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=5)

    im_dim, sens_dim = trainset.__get_sens_dim__()
    print('done')

    # load network
    print('\nloading the network ...\n')
    # (linear attn) insert attention befroe or after maxpooling?
    # (grid attn only supports "before" mode)
    print('\npay attention after maxpooling layers...\n')
    model = AttnVGG_sensor_mask(im_size=im_dim, sens_dim=sens_dim, attention=True, normalize_attn=True, init='xavierUniform').cuda()

    criterion = nn.MSELoss()
    print('done')

    """
    # move to GPU
    print('\nmoving to GPU ...\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    print('done')
    """

    # SGD optimizer
    """
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    """
    optimizer = RAdam(model.parameters())

    if train:
        # training
        print('\nstart training ...\n')
        step = 0
        running_avg_accuracy = 0
        writer = SummaryWriter(opt.outf)
        for epoch in range(opt.epochs):
            images_disp = []
            train_loss = []
            # adjust learning rate
            """
            scheduler.step()
            """
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            print("\nepoch %d learning rate %f\n" % (epoch, optimizer.param_groups[0]['lr']))
            # run for one epoch
            for i, data in enumerate(trainloader, 0):
                # warm up
                model.train()
                model.zero_grad()

                optimizer.zero_grad()

                im_batch, sens_batch, act_batch = data

                mini_size = 4
                batch_break = int(opt.batch_size / mini_size)

                for idx in range(batch_break):
                    #print(i, batch_break, i * batch_break, (i + 1) * batch_break)
                    im, sens, act = im_batch[idx * mini_size: (idx + 1) * mini_size].cuda(), sens_batch[idx * mini_size: (idx + 1)
                                                                                                            * mini_size].cuda(), act_batch[idx * mini_size: (idx + 1) * mini_size].cuda()
                    # forward
                    pred, __, __, __, __ = model(im, sens)
                    # backward
                    loss = criterion(pred, act) / mini_size
                    loss.backward()

                optimizer.step()
                # display results
                train_loss.append(loss.item())
                writer.add_scalar('train/loss', loss.item(), step)

                print("[epoch %d][%d/%d] loss %.4f " % (epoch, i, len(trainloader) - 1, loss.item() * 4))
                step += 1

            print("[epoch %d] loss %.4f " % (epoch, np.mean(train_loss) * 4))
            # the end of each epoch: test & log
            print('\none epoch done, saving records ...\n')
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
            if epoch == opt.epochs / 100:
                torch.save(model.state_dict(), os.path.join(opt.outf, 'net%d.pth' % epoch))

        else:
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                # log scalars
                for i, data in enumerate(testloader, 0):
                    images_test, labels_test = data
                    images_test, labels_test = images_test.to(device), labels_test.to(device)
                    if i == 0:  # archive images in order to save to logs
                        images_disp.append(inputs[0:36, :, :, :])
                    pred_test, __, __, __ = model(images_test)
                    predict = torch.argmax(pred_test, 1)
                    total += labels_test.size(0)
                    correct += torch.eq(predict, labels_test).sum().double().item()
                writer.add_scalar('test/accuracy', correct / total, epoch)
                print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100 * correct / total))
                # log images
                if opt.log_images:
                    print('\nlog images ...\n')
                    I_train = utils.make_grid(images_disp[0], nrow=6, normalize=True, scale_each=True)
                    writer.add_image('train/image', I_train, epoch)
                    if epoch == 0:
                        I_test = utils.make_grid(images_disp[1], nrow=6, normalize=True, scale_each=True)
                        writer.add_image('test/image', I_test, epoch)
                if opt.log_images and (not opt.no_attention):
                    print('\nlog attention maps ...\n')
                    # base factor
                    min_up_factor = 2
                    # sigmoid or softmax
                    if opt.normalize_attn:
                        vis_fun = visualize_attn_softmax
                    else:
                        vis_fun = visualize_attn_sigmoid
                    # training data
                    __, c1, c2, c3 = model(images_disp[0])
                    if c1 is not None:
                        attn1 = vis_fun(I_train, c1, up_factor=min_up_factor, nrow=6)
                        writer.add_image('train/attention_map_1', attn1, epoch)
                    if c2 is not None:
                        attn2 = vis_fun(I_train, c2, up_factor=min_up_factor * 2, nrow=6)
                        writer.add_image('train/attention_map_2', attn2, epoch)
                    if c3 is not None:
                        attn3 = vis_fun(I_train, c3, up_factor=min_up_factor * 4, nrow=6)
                        writer.add_image('train/attention_map_3', attn3, epoch)
                    # test data
                    __, c1, c2, c3 = model(images_disp[1])
                    if c1 is not None:
                        attn1 = vis_fun(I_test, c1, up_factor=min_up_factor, nrow=6)
                        writer.add_image('test/attention_map_1', attn1, epoch)
                    if c2 is not None:
                        attn2 = vis_fun(I_test, c2, up_factor=min_up_factor * 2, nrow=6)
                        writer.add_image('test/attention_map_2', attn2, epoch)
                    if c3 is not None:
                        attn3 = vis_fun(I_test, c3, up_factor=min_up_factor * 4, nrow=6)
                        writer.add_image('test/attention_map_3', attn3, epoch)

if __name__ == "__main__":
    main()
