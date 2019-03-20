"""OKU19 Dataset Classes

"""

import os
import csv
import os.path
import torch
import torch.utils.data as data
import cv2, pickle
import numpy as np

# CLASSES = (  # always index 0
#         'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',
#         'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
#         'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
#         'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')

# CLASSES = (  # always index 0
#         'Calling', 'Carrying', 'Drinking', '"Hand', 'Hugging', 'Lying', 'Pushing/Pulling',
#          'Reading', 'Running', 'Sitting', 'Standing', 'Walking')

CLASSES = (  # always index 0
        'Person')

# CLASSES = (  # always index 0
#         'Person',)


class AnnotationTransform(object):
    """
    Same as original
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of the datasets classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))
        # self.ind_to_class = dict(zip(range(len(CLASSES)),CLASSES))

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]

        oku19 target [  0 Track ID. please check the below part for details
                        1 xmin. The top left x-coordinate of the bounding box.
                        2 ymin. The top left y-coordinate of the bounding box.
                        3 xmax. The bottom right x-coordinate of the bounding box.
                        4 ymax. The bottom right y-coordinate of the bounding box.
                        5 frame. The frame that this annotation represents.
                        6 lost. If 1, the annotation is outside of the view screen.
                        7 occluded. If 1, the annotation is occluded.
                        8 generated. If 1, the annotation was automatically interpolated.
                        9 label. The label for this annotation, enclosed in quotation marks. This field is always “Person”.
                        10 (+) actions. Each column after this is an action.]
        """

        res = []
        for t in target:
            if t[6] == '0':
                if t[7] == '0':
                    if t[9] != '':
                        pts = [t[1], t[2], t[3], t[4]]
                        '''pts = ['xmin', 'ymin', 'xmax', 'ymax']'''
                        bndbox = []
                        for i in range(4):
                            cur_pt = max(0,int(pts[i]) - 1)
                            scale =  width if i % 2 == 0 else height
                            cur_pt = min(scale, int(pts[i]))
                            cur_pt = float(cur_pt) / scale
                            bndbox.append(cur_pt)
                            print(t[9])
                            print(self.class_to_ind[t[9]])
                        label_idx = self.class_to_ind[t[9]]
                        bndbox.append(label_idx)
                        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class OKU19Detection(data.Dataset):
    """OKU19 Action Detection Dataset
    to access input images and target which is annotation

    input is image, target is annotation

    Arguments:
        root (string): filepath to OKU19 folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_set, transform=None, target_transform=AnnotationTransform(),
                 dataset_name='oku19', input_type='rgb', full_test=False):

        self.input_type = input_type
        input_type = input_type+'-images'
        self.root = root
        self.CLASSES = CLASSES
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(root, image_set+'-Set/Labels/SingleActionTrackingLabels' + '%s.csv')
        self._imgpath = os.path.join(root, image_set+'-Set', input_type + '%s.jpg')
        self._negpath = os.path.join(root, '%s.csv')
        # print("annopath: " + self._annopath)
        # print("imgpath: " + self._imgpath)
        self.ids = list()
        # root = /vol/guy/oku19/1280x720
        for line in open(os.path.join(root, 'splitfiles', image_set + 'valstandard.txt')):
            self.ids.append(line.strip())

        # if self.image_set == 'train':
        #     self.ids = trainlist
        # elif self.image_set == 'test':
        #     self.ids = testlist
        # else:
        #     print('spacify correct subset ')

        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

# TODO: make list has been removed.. make train test video lists another way..................
        # trainlist, testlist, video_list = make_lists(root, input_type, split=1, fulltest=full_test)
        # self.video_list = video_list
        # if self.image_set == 'train':
        #     self.ids = trainlist
        # elif self.image_set == 'test':
        #     self.ids = testlist
        # else:
        #     print('spacify correct subset ')


    def __getitem__(self, index):
        im, gt, img_index = self.pull_item(index)

        return im, gt, img_index

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        # needs to open csv file
        target = []
        if os.path.isfile(self._annopath % img_id):
            with open(self._annopath % img_id, 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in spamreader:
                    target.append(row)
        else:
            with open(self._negpath % 'negative', 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in spamreader:
                    target.append(row)


#        with open(self._annopath % img_id, 'r') as csvfile:
#            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#            for row in spamreader:
#                target.append(row)
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # print(height, width,target)
        return torch.from_numpy(img).permute(2, 0, 1), target, index
        # return torch.from_numpy(img), target, height, width


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """

    targets = []
    imgs = []
    image_ids = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2])
    return torch.stack(imgs, 0), targets, image_ids
