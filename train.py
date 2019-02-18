# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data

from data import AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from data import v2, OKU19Detection, AnnotationTransform, detection_collate, CLASSES, BaseTransform
from data import KittiLoader, AnnotationTransform_kitti,Class_to_ind

from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from IPython import embed
from log import log
import time

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--dim', default=512, type=int, help='Size of the input image, only support 300 or 512')
parser.add_argument('-d', '--dataset', default='VOC',help='VOC or COCO dataset')

parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--stepvalues', default='30000,60000,100000', type=str, help='iter numbers where learing rate to be dropped')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--save_folder', default='weights/', help='Location to save ch    args.stepvalues = [int(val) for val in args.stepvalues.split(',')]eckpoint models')
parser.add_argument('--data_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# train_sets = 'train'
# means = (104, 117, 123)  # only support voc now
# if args.dataset=='VOC':
#     num_classes = len(VOC_CLASSES) + 1
#     train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# elif args.dataset=='oku19':
#     num_classes = len(CLASSES) + 1
# elif args.dataset=='oku19':
#     num_classes = len(CLASSES) + 1
# elif args.dataset=='kitti':
#     num_classes = 1+1
# accum_batch_size = 32
# iter_size = accum_batch_size / args.batch_size
# start_iter = 0
# tepvalues = [int(val) for val in args.stepvalues.split(',')]
# print_step = 10
# loss_reset_step = 30
#
#     ## Define the experiment Name will used to same directory and ENV for visdom
# exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.dataset,
#                 args.input_type, args.batch_size, args.basenet[:-14], int(args.lr*100000))
#
# save_root += args.dataset+'/'
# save_root = args.save_root+'cache/'+args.exp_name+'/'
#
# if args.visdom:
#     import visdom
#     viz = visdom.Visdom()
#
# ssd_net = build_ssd('train', args.dim, num_classes)
# net = ssd_net
#
# if args.cuda:
#     net = torch.nn.DataParallel(ssd_net)
#     cudnn.benchmark = True
#
# if args.resume:
#     log.l.info('Resuming training, loading {}...'.format(args.resume))
#     ssd_net.load_weights(args.resume)
#     start_iter = int(agrs.resume.split('/')[-1].split('.')[0].split('_')[-1])
# else:
#     vgg_weights = torch.load(args.save_folder + args.basenet)
#     log.l.info('Loading base network...')
#     ssd_net.vgg.load_state_dict(vgg_weights)
#     start_iter = 0


#
# if not args.resume:
#     log.l.info('Initializing weights...')
#     # initialize newly added layers' weights with xavier method
#     ssd_net.extras.apply(weights_init)
#     ssd_net.loc.apply(weights_init)
#     ssd_net.conf.apply(weights_init)
#
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)
# criterion = MultiBoxLoss(num_classes, args.dim, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

def main():
    args.cfg = v2
    args.train_sets = 'train'
    args.means = (104, 117, 123)
    num_classes = len(CLASSES) + 1
    args.num_classes = num_classes
    args.stepvalues = [int(val) for val in args.stepvalues.split(',')]
    args.loss_reset_step = 30
    args.eval_step = 10000
    args.print_step = 10

    ## Define the experiment Name will used to same directory and ENV for visdom
    args.exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.dataset,
                args.input_type, args.batch_size, args.basenet[:-14], int(args.lr*100000))

    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    net = build_ssd(300, args.num_classes)

    if args.cuda:
        net = net.cuda()
    # if args.cuda:
    #     net = torch.nn.DataParallel(ssd_net)
    #     cudnn.benchmark = True

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()

    print('Initializing weights for extra layers and HEADs...')
    # initialize newly added layers' weights with xavier method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    if args.input_type == 'fastOF':
        print('Download pretrained brox flow trained model weights and place them at:::=> ',args.data_root + 'ucf24/train_data/brox_wieghts.pth')
        pretrained_weights = args.data_root + 'ucf24/train_data/brox_wieghts.pth'
        print('Loading base network...')
        net.load_state_dict(torch.load(pretrained_weights))
    else:
        vgg_weights = torch.load(args.data_root +'ucf24/train_data/' + args.basenet)
        print('Loading base network...')
        net.vgg.load_state_dict(vgg_weights)

    args.data_root += args.dataset + '/'

    parameter_dict = dict(net.named_parameters()) # Get parmeter of network in dictionary format wtih name being key
    params = []

    #Set different learning rate to bias layers and set their weight_decay to 0
    for name, param in parameter_dict.items():
        if name.find('bias') > -1:
            print(name, 'layer parameters will be trained @ {}'.format(args.lr*2))
            params += [{'params': [param], 'lr': args.lr*2, 'weight_decay': 0}]
        else:
            print(name, 'layer parameters will be trained @ {}'.format(args.lr))
            params += [{'params':[param], 'lr': args.lr, 'weight_decay':args.weight_decay}]

    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = MultiBoxLoss(args.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    scheduler = MultiStepLR(optimizer, milestones=args.stepvalues, gamma=args.gamma)
    train(args, net, optimizer, criterion, scheduler)


def DatasetSync(dataset='VOC',split='training'):


    if dataset=='VOC':
        #DataRoot=os.path.join(args.data_root,'VOCdevkit')
        DataRoot=args.data_root
        dataset = VOCDetection(DataRoot, train_sets, SSDAugmentation(
        args.dim, means), AnnotationTransform())
    elif dataset=='oku19':
        dataset = OKU19Detection(args.data_root, args.train_sets, SSDAugmentation(args.ssd_dim, args.means),
                                           AnnotationTransform(), input_type=args.input_type)
    elif dataset=='kitti':
        DataRoot=os.path.join(args.data_root,'kitti')
        dataset = KittiLoader(DataRoot, split=split,img_size=(1000,300),
                  transforms=SSDAugmentation((1000,300),means),
                  target_transform=AnnotationTransform_kitti())
    return dataset

def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    log.l.info('Loading Dataset...')

    # dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
    #     args.dim, means), AnnotationTransform())
    dataset=DatasetSync(dataset=args.dataset,split='training')


    epoch_size = len(dataset) // args.batch_size
    log.l.info('Training SSD on {}'.format(dataset.name))
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)

    lr=args.lr
    for iteration in range(start_iter, args.iterations + 1):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            lr=adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)
        #embed()
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 10 == 0:
            log.l.info('''
                Timer: {:.5f} sec.\t LR: {}.\t Iter: {}.\t Loss_l: {:.5f}.\t Loss_c: {:.5f}.
                '''.format((t1-t0),lr,iteration,loss_l.data[0],loss_c.data[0]))
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        if iteration % 5000 == 0:
            log.l.info('Saving state, iter: {}'.format(iteration))
            torch.save(ssd_net.state_dict(), 'weights/ssd' + str(args.dim) + '_0712_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + 'ssd_' + str(args.dim) + '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
