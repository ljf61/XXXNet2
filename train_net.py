import sys
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from net import VONet
from data import MyDataset
#import pytorch_ssim
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)

def train_net(net, epochs=5, batchSize=8, lr=0.001, val_percent=0.05,
              cp=True, gpu=True, alpha=0.5):
    directory = '/home/ljf/LJF/XXXNET2/test6/data/'
    datasetImage = 'rgb_list.txt'
    datasetDepth = 'depth_list.txt'
    dir_checkpoint = '/home/ljf/LJF/XXXNET2/test6/weights/'
    datasetPose = 'vo_gt.txt'
    poses = []
    loss_record = []
    images = []
    depths= []
    with open(directory+datasetImage) as f1:
        next(f1)  # skip the 3 header lines
        for line in f1:
            fname = line.strip('\n')
            images.append(directory+fname+'.png')
    f1.close()
    with open(directory+datasetDepth) as f2:
        next(f2)  # skip the 3 header lines
        for line in f2:
            fname = line.strip('\n')
            depths.append(directory+fname+'.png')
    f2.close()
    with open(directory + datasetPose) as f3:
        next(f3)  # skip the 3 header lines
        for line in f3:
            p0, p1, p2, p3, p4, p5, p6= line.split()
            #p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            poses.append((p1, p2))
            #poses.append((p1, p2, p3, p4, p5, p6))
            #poses_yaw.append((p5, p6))
    f3.close()


    my_dataset = MyDataset(poses,images,depths)

    train_loader = torch.utils.data.DataLoader(dataset=my_dataset,
                                               batch_size=batchSize,
                                               shuffle=True,
                                               )
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
    criterion = nn.SmoothL1Loss()
    #ssim_loss = pytorch_ssim.SSIM(window_size = 200)
    #criterion2 = nn.MSELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch, epochs))
        l1 = 0;
        l2 = 0;
        l3 = 0;
        for i, (pose_gt, images1, images2, depth_gt) in enumerate(train_loader):
            images1 = Variable(images1).cuda()
            images2 = Variable(images2).cuda()
            depth_gt = Variable(depth_gt).cuda()
            pose_gt = Variable(pose_gt).cuda() 

            optimizer.zero_grad() 

            depth, pose = net(images1,images2)
            #loss1 = criterion(depth, depth_gt)+(1-ssim_loss(depth, depth_gt))*50
            loss1 = criterion(depth, depth_gt)
            l1 += loss1.data[0]
            loss2 = criterion(pose, pose_gt)*100
            l2 += loss2.data[0]
            loss = loss1+loss2
            l3 += loss.data[0]
            loss.backward()           
            optimizer.step()
            i = i + 1
            if i%50 == 0:
                print ('Epoch [%d/%d], Step [%d], ave_Loss1: %.4f, ave_Loss2: %.4f,Loss: %.4f'
                       %(epoch+1, epochs, i+1,  l1/i, l2/i, l3/i))
                loss_record.append((l1/i, l2/i, l3/i))
        if cp and epoch%10 == 0:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}_0.pth'.format(epoch))

            print('Checkpoint {} saved !'.format(epoch + 1))

    loss_record = np.array(loss_record)
    np.savetxt('/home/ljf/LJF/XXXNET2/test5/loss_record.txt',loss_record, delimiter=' ')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=301, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=20,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()

    net = VONet()
    net.apply(weights_init)

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        train_net(net, options.epochs, options.batchsize, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        sys.exit(0)

