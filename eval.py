"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOCroot, UCF24root, OKU19root
from data import UCF24CLASSES as labelmap
import torch.utils.data as data
from utils.evaluation import evaluate_detections
from layers.box_utils import decode, nms
# from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import UCF24Detection, AnnotationTransform, detection_collate, UCF24CLASSES, BaseTransform
from IPython import embed
from ssd import build_ssd
from log import log
from data import v
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd512_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--data_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('-d', '--dataset', default='VOC',help='VOC or COCO dataset')
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb can take flow as well')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--ssd_dim', default=512, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.data_root, args.dataset, 'Annotations', '%s.xml')
imgpath = os.path.join(args.data_root, args.dataset, 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.data_root, args.dataset, 'ImageSets', 'Main', '{:s}.txt')

devkit_path = UCF24root + 'UCF24'
dataset_mean = (104, 117, 123)
set_type = 'test'

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=512, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd512_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        log.l.info('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    log.l.info('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def validate(args, net, val_data_loader, val_dataset, iteration_num, iou_thresh=0.5):
    """Test a SSD network on an image database."""
    print('Validating at ', iteration_num)
    num_images = len(val_dataset)
    num_classes = args.num_classes

    det_boxes = [[] for _ in range(len(UCF24CLASSES))]
    gt_boxes = []
    print_time = True
    batch_iterator = None
    val_step = 100
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()

    for val_itr in range(len(val_data_loader)):
        if not batch_iterator:
            batch_iterator = iter(val_data_loader)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        images, targets, img_indexs = next(batch_iterator)
        batch_size = images.size(0)
        height, width = images.size(2), images.size(3)

        if args.cuda:
            images = Variable(images.cuda(), volatile=True)
        output = net(images)

        loc_data = output[0]
        conf_preds = output[1]
        prior_data = output[2]

        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            tf = time.perf_counter()
            print('Forward Time {:0.3f}'.format(tf-t1))
        for b in range(batch_size):
            gt = targets[b].numpy()
            gt[:,0] *= width
            gt[:,2] *= width
            gt[:,1] *= height
            gt[:,3] *= height
            gt_boxes.append(gt)
            decoded_boxes = decode(loc_data[b].data, prior_data.data, args.cfg['variance']).clone()
            conf_scores = net.softmax(conf_preds[b]).data.clone()

            for cl_ind in range(1, num_classes):
                scores = conf_scores[:, cl_ind].squeeze()
                c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                scores = scores[c_mask].squeeze()
                # print('scores size',scores.size())
                if scores.dim() == 0:
                    # print(len(''), ' dim ==0 ')
                    det_boxes[cl_ind - 1].append(np.asarray([]))
                    continue
                boxes = decoded_boxes.clone()
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes = boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                scores = scores[ids[:counts]].cpu().numpy()
                boxes = boxes[ids[:counts]].cpu().numpy()
                # print('boxes sahpe',boxes.shape)
                boxes[:,0] *= width
                boxes[:,2] *= width
                boxes[:,1] *= height
                boxes[:,3] *= height

                for ik in range(boxes.shape[0]):
                    boxes[ik, 0] = max(0, boxes[ik, 0])
                    boxes[ik, 2] = min(width, boxes[ik, 2])
                    boxes[ik, 1] = max(0, boxes[ik, 1])
                    boxes[ik, 3] = min(height, boxes[ik, 3])

                cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)

                det_boxes[cl_ind-1].append(cls_dets)
            count += 1
        if val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
            torch.cuda.synchronize()
            ts = time.perf_counter()
        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('NMS stuff Time {:0.3f}'.format(te - tf))
    print('Evaluating detections for itration number ', iteration_num)
    return evaluate_detections(gt_boxes, det_boxes, UCF24CLASSES, iou_thresh=iou_thresh)


if __name__ == '__main__':
    # load net
    args.cfg = v[str(args.ssd_dim)]
    num_classes = len(UCF24CLASSES) + 1 # +1 background
    args.num_classes = num_classes
    net = build_ssd('test', 512, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    log.l.info('Finished loading model!')
    # load data
    val_dataset = UCF24Detection(args.data_root, set_type, BaseTransform(512, dataset_mean), AnnotationTransform(), input_type=args.input_type, full_test=False)
    val_data_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, collate_fn=detection_collate, pin_memory=True)
    # val_dataset = UCF24Detection(args.data_root, 'train', BaseTransform(args.ssd_dim, args.means), AnnotationTransform(), input_type=args.input_type, full_test=False)
    # dataset = VOCDetection(args.data_root, [('2007', set_type)], BaseTransform(512, dataset_mean), AnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    # validate(+++++++++++++++++++++++++++++++++++++++++++++++++++++++++)
    mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, 0, iou_thresh=args.iou_thresh)
    for ap_str in ap_strs:
        log.l.info(ap_str+'\n')
    ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
    log.l.info(ptr_str)

    # test_net(args.save_folder, net, args.cuda, dataset,
    #          BaseTransform(net.size, dataset_mean), args.top_k, 512,
    #          thresh=args.confidence_threshold)


    #
    # torch.cuda.synchronize()
    # tvs = time.perf_counter()
    # log.l.info('Saving state, iter: {}'.format(iteration))
    # torch.save(ssd_net.state_dict(), save_folder + 'weights/ssd' + str(args.dim) + '_0712_' +
    #            repr(iteration) + '.pth')
    #
    # net.eval() # switch net to evaluation mode
    # mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)
    #
    # for ap_str in ap_strs:
    #     print(ap_str)
    #     log.l.info(ap_str+'\n')
    # ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
    # log.l.info(ptr_str)
    #
    # if args.visdom:
    #     aps = [mAP]
    #     for ap in ap_all:
    #         aps.append(ap)
    #     viz.line(
    #         X=torch.ones((1, args.num_classes)).cpu() * iteration,
    #         Y=torch.from_numpy(np.asarray(aps)).unsqueeze(0).cpu(),
    #         win=val_lot,
    #         update='append'
    #             )
    # net.train() # Switch net back to training mode
    # torch.cuda.synchronize()
    # t0 = time.perf_counter()
    # prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
    # log.l.info(ptr_str)
