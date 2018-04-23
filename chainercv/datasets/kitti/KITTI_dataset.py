import glob
import os
from urllib.parse import urljoin

import numpy as np

import chainer
from chainer.dataset import download
from chainercv import utils
from chainercv.utils.image import read_image

# root = 'pfnet/chainercv/KITTI'
url_base = 'http://kitti.is.tue.mpg.de/kitti/raw_data/'

# use pykitti
import pykitti

import itertools
# tracklet parser
from chainercv.datasets.kitti import parseTrackletXML as xmlParser

# check 
import matplotlib.pyplot as plt
from chainercv.visualizations import vis_bbox

# image_shape = 375, 1242

KITTI_category_names = (
    'City',
    'Residential',
    'Road',
    'Campus',
    'Person',
    'Calibration'
)

KITTI_label_names = (
    'Car', 
    'Van', 
    'Truck', 
    'Pedestrian',
    'Sitter', 
    'Cyclist',
    'Tram', 
    'Misc',
)

KITTI_label_colors = (
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (60, 40, 222),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
)
KITTI_ignore_label_color = (0, 0, 0)


class KITTIDataset(chainer.dataset.DatasetMixin):

    """Image dataset for test split of `KITTI dataset`_.

    .. _`KITTI dataset`: http://www.cvlibs.net/datasets/kitti/raw_data.php

    .. note::

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain the :obj:`---` directory. If :obj:`auto` is given,
            it uses :obj:`$CHAINER_DATSET_ROOT/pfnet/chainercv/KITTI` by
            default.

    """

    def __init__(self, data_dir='auto', date='', driveNo='', color=True, sync=True, isLeft=True):
        self.color = color
        self.sync = sync
        self.isLeft = isLeft
        if data_dir == 'auto':
            if sync == True:
                # download sync data
                # data_dir = self.get_KITTI_Sync_Data('pfnet/chainercv/KITTI', date, driveNo)
                data_dir = self.get_KITTI_Sync_Data(os.path.join('pfnet', 'chainercv', 'KITTI'), date, driveNo)
            else:
                # download nosync data
                # data_dir = self.get_KITTI_NoSync_Data('pfnet/chainercv/KITTI', date, driveNo)
                data_dir = self.get_KITTI_NoSync_Data(os.path.join('pfnet', 'chainercv', 'KITTI'), date, driveNo)

        # no use pykitti
        # if self.color == True:
        #     if self.isLeft == True:
        #         imgNo = '02'
        #     else:
        #         imgNo = '03'
        # else:
        #     if self.isLeft == True:
        #         imgNo = '00'
        #     else:
        #         imgNo = '01'
        # image
        # self.get_KITTI_Image(data_dir, date, driveNo, imgNo)
        # img = read_image(self.img_paths[0])
        ##
        # use pykitti
        # read All images
        # imformat='None'
        # self.dataset = pykitti.raw(data_dir, date, driveNo, frames=None)
        self.dataset = pykitti.raw(data_dir, date, driveNo, frames=None, imformat='cv2')

        # current camera calibration R/P settings.
        if self.color == True:
            if self.isLeft == True:
                # img02
                self.cur_R_rect = self.dataset.calib.R_rect_20
                self.cur_P_rect = self.dataset.calib.P_rect_20
                self.imgs = np.array(list(self.dataset.cam2))
                # img = np.array(list(self.dataset.rgb)[0])
                # img = np.uint8(np.array(list(self.dataset.rgb)[0]) * 255).astype(np.float32)
            else:
                # img03
                self.cur_R_rect = self.dataset.calib.R_rect_30
                self.cur_P_rect = self.dataset.calib.P_rect_30
                self.imgs = np.array(list(self.dataset.cam3))
                # img = np.array(list(self.dataset.rgb)[1])
                # img = np.uint8(np.array(list(self.dataset.rgb)[1]) * 255).astype(np.float32)
        else:
            if self.isLeft == True:
                # img00
                self.cur_R_rect = self.dataset.calib.R_rect_00
                self.cur_P_rect = self.dataset.calib.P_rect_00
                self.imgs = np.array(list(self.dataset.cam0))
                # img = np.array(list(self.dataset.gray)[0])
            else:
                # img01
                self.cur_R_rect = self.dataset.calib.R_rect_10
                self.cur_P_rect = self.dataset.calib.P_rect_10
                self.imgs = np.array(list(self.dataset.cam1))
                # img = np.array(list(self.dataset.gray)[1])

        # get object info(type/area/...)
        self.tracklets = self.get_KITTI_Tracklets(data_dir, date, driveNo)

        # set arrays
        # length = len(self.dataset.cam0)
        length = self.__len__()
        self.bboxes = [0] * length
        self.labels = [0] * length
        for idx in range(0, length):
            self.bboxes[idx] = list()
            self.labels[idx] = list()

        self.get_KITTI_Label(self.tracklets)


    def __getitem__(self, index):
        # Note : before setting datas.
        # no use pykitti
        # return read_image(self.img_paths[index])
        # use pykitti
        img = self.imgs[index]
        bbox = self.bboxes[index]
        label = self.labels[index]

        # convert data is utils.read_image function return values
        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            # return img[np.newaxis]
            return img[np.newaxis], bbox, label
        else:
            # transpose (H, W, C) -> (C, H, W)
            return img.transpose((2, 0, 1)), bbox, label


    def __len__(self):
        # no use pykitti
        # return len(self.img_paths)
        # use pykitti
        # print(self.imgs)
        return len(self.imgs)


#     # img
#     def get_example(self, i):
#         """Returns the i-th test image.
# 
#         Returns a color image. The color image is in CHW format.
# 
#         Args:
#             i (int): The index of the example.
# 
#         Returns:
#             A color image whose shape is (3, H, W). H and W are height and
#             width of the image.
#             The dtype of the color image is :obj:`numpy.float32`.
# 
#         """
#         # return read_image(self.img_paths[i])
###

#      def get_example(self, i):
#         """Returns the i-th example.
# 
#         Returns a color image and a label image. The color image is in CHW
#         format and the label image is in HW format.
# 
#         Args:
#             i (int): The index of the example.
# 
#         Returns:
#             tuple of a color image and a label whose shapes are (3, H, W) and
#             (H, W) respectively. H and W are height and width of the image.
#             The dtype of the color image is :obj:`numpy.float32` and
#             the dtype of the label image is :obj:`numpy.int32`.
#         """
#         if i >= len(self):
#             raise IndexError('index is too large')
#         img_path, label_path = self.paths[i]
#         img = read_image(img_path, color=True)
#         label = read_image(label_path, dtype=np.int32, color=False)[0]
#         # Label id 11 is for unlabeled pixels.
#         label[label == 11] = -1
#         return img, label
# 
###

    # point
    def get_example(self, i):
        """Returns the i-th test point.

        Returns a color point. The color point is in XYZRGB format.

        Args:
            i (int): The index of the example.

        Returns:
            A color point whose shape is (6, x, y, z). H and W are height and
            width of the point.
            The dtype of the color point is :obj:`numpy.float32`.

        """
        # use pykitti
        # return self.dataset.gray[i]
        # return next(iter(itertools.islice(self.dataset.velo, 0, None)))
        label = self.labels[i]
        # return next(iter(itertools.islice(self.dataset.velo, i, None)))
        return next(iter(itertools.islice(self.dataset.velo, i, None))), bbox, label


    def get_KITTI_Sync_Data(self, root, date, driveNo):
        data_root = download.get_dataset_directory(root)
        # print('dst path : ' + data_root)

        # data
        folder = date + '_drive_' + driveNo
        # ok
        # url_data = url_base + '/' +  folder+ '/' + folder + '_sync.zip'
        url_data = urljoin(url_base, folder+ '/' + folder + '_sync.zip')
        # print('url : ' + url_data)

        # calibration
        url_calib = url_base + date + '_calib.zip'

        # tracklet
        url_tracklet = urljoin(url_base, folder+ '/' + folder + '_tracklets.zip')

        download_file_path = utils.cached_download(url_data)
        ext = os.path.splitext(url_data)[1]
        utils.extractall(download_file_path, data_root, ext)

        download_file_path = utils.cached_download(url_calib)
        ext = os.path.splitext(url_calib)[1]
        utils.extractall(download_file_path, data_root, ext)

        download_file_path = utils.cached_download(url_tracklet)
        ext = os.path.splitext(url_tracklet)[1]
        utils.extractall(download_file_path, data_root, ext)

        return data_root


    def get_KITTI_NoSync_Data(self, root, date, driveNo):
        data_root = download.get_dataset_directory(root)

        # data
        folder = date + '_drive_' + driveNo
        url_data = urljoin(url_base, folder+ '/' + folder + '_extract.zip')

        # calibration
        url_calib = url_base + date + '_calib.zip'

        # tracklet
        url_tracklet = urljoin(url_base, folder+ '/' + folder + '_tracklets.zip')

        download_file_path = utils.cached_download(url_data)
        ext = os.path.splitext(url_data)[1]
        utils.extractall(download_file_path, data_root, ext)

        download_file_path = utils.cached_download(url_calib)
        ext = os.path.splitext(url_calib)[1]
        utils.extractall(download_file_path, data_root, ext)

        download_file_path = utils.cached_download(url_tracklet)
        ext = os.path.splitext(url_tracklet)[1]
        utils.extractall(download_file_path, data_root, ext)

        return data_root


    def get_KITTI_Image(self, data_root, date, driveNo, ImgNo):
        """
        no use pykitti
        """
        folder = date + '_drive_' + driveNo
        img_dir = os.path.join(data_root, os.path.join(date, folder + '_sync', 'image_' + ImgNo, 'data'))

        self.img_paths = list()
        if not os.path.exists(img_dir):
            raise ValueError(
                'KITTI dataset does not exist at the expected location.'
                'Please download it from http://www.cvlibs.net/datasets/kitti/raw_data.php.'
                'Then place directory image at {}.'.format(img_dir))

        for img_path in sorted(glob.glob(os.path.join(img_dir, '*.png'))):
                self.img_paths.append(img_path)


    def get_KITTI_Tracklets(self, data_root, date, driveNo):
        # read calibration files
        kitti_dir = os.path.join(data_root, date)
        # kitti_dir = kitti_dir.replace(os.path.sep, '/')
        # calibration_dir = os.path.join(data_root, date)
        # self.imu2velo = read_calib_file(os.path.join(kitti_dir, "calib_imu_to_velo.txt"))
        # self.velo2cam = read_calib_file(os.path.join(kitti_dir, "calib_velo_to_cam.txt"))
        # self.cam2cam = read_calib_file(os.path.join(kitti_dir, "calib_cam_to_cam.txt"))
        # read tracklet
        folder = date + '_drive_' + driveNo + '_sync'
        # self.tracklet = read_tracklet_file(os.path.join(kitti_dir, folder, "calib_imu_to_velo.txt"))
        # return tracklets
        # get dir names
        # read tracklets from file
        myTrackletFile = os.path.join(kitti_dir, folder, 'tracklet_labels.xml')
        tracklets = xmlParser.parseXML(myTrackletFile)
        return tracklets


    def get_KITTI_Label(self, tracklets):
        twoPi = 2.*np.pi
        # loop over tracklets
        for iTracklet, tracklet in enumerate(tracklets):
            # print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

            # this part is inspired by kitti object development kit matlab code: computeBox3D
            h,w,l = tracklet.size
            trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
            [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

            # print('trackletBox : ' + trackletBox)
            # print(trackletBox)
            objTypeStr = tracklet.objectType
            # print(objTypeStr)

            # loop over all data in tracklet
            for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber \
                in tracklet:

                # determine if object is in the image; otherwise continue
                if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                    continue

                # re-create 3D bounding box in velodyne coordinate system
                yaw = rotation[2]   # other rotations are 0 in all xml files I checked
                assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
                rotMat = np.array([\
                            [np.cos(yaw), -np.sin(yaw), 0.0], \
                            [np.sin(yaw),  np.cos(yaw), 0.0], \
                            [        0.0,          0.0, 1.0]])
                cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T
                # print(cornerPosInVelo)
                # print(cornerPosInVelo[:,0])
                # print(cornerPosInVelo.shape) 3*8
                # print('tracklet for : ')
                # print(iTracklet)
                # print(translation)
                # print(rotation)
                # print(state)
                # print(occlusion)
                # print(truncation)
                # print(amtOcclusion)
                # print(amtBorders)

                # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to 
                #   car-centered yaw (i.e. 0 degree = same orientation as car).
                #   makes quite a difference for objects in periphery!
                # Result is in [0, 2pi]
                x, y, z = translation
                # print(translation)
                yawVisual = ( yaw - np.arctan2(y, x) ) % twoPi
                # print(yaw)
                # print(yawVisual)
                # param = pykitti.utils.transform_from_rot_trans(rotMat, translation)
                # print(param)

                # projection to image?
                # print(self.dataset.calib.P_rect_20)
                # param3 = translation.reshape(3, 1) * self.dataset.calib.P_rect_20
                # print(cornerPosInVelo[:, 0:1].shape)
                pt3d = np.vstack((cornerPosInVelo[:,0:8], np.ones(8)))
                # print(pt3d.shape)
                # print(self.dataset.calib.P_rect_20)

                pt2d = self.project_velo_points_in_img(pt3d, self.dataset.calib.T_cam2_velo, self.cur_R_rect, self.cur_P_rect)

                # print(pt2d)
                xmin = min(pt2d[0, :])
                xmax = max(pt2d[0, :])
                ymin = min(pt2d[1, :])
                ymax = max(pt2d[1, :])
                param = np.array((ymin, xmin, ymax, xmax))
                # bbox.append(param)
                # bbox = np.stack(bbox).astype(np.float32)
                # self.bboxes[absoluteFrameNumber] = bbox
                self.bboxes[absoluteFrameNumber].append(param)
                # print(self.bboxes[absoluteFrameNumber])
                # param_3d = cornerPosInVelo
                # self.bboxes_3d[absoluteFrameNumber].append(cornerPosInVelo)
                # label.append(param2)
                # label = np.stack(label).astype(np.int32)
                # self.labels[absoluteFrameNumber] = label
                # objectType
                # label_names
                # not search objTypeStr? process
                param2 = KITTI_label_names.index(objTypeStr)
                self.labels[absoluteFrameNumber].append(param2)
                # print(self.bboxes[absoluteFrameNumber])

            # end : for all frames in track
        # end : for all tracks

    # def get_KITTI_Calibration(self, data_root, date, driveNo):

    def project_velo_points_in_img(self, pts3d, T_cam_velo, Rrect, Prect):
        """Project 3D points into 2D image. Expects pts3d as a 4xN
           numpy array. Returns the 2D projection of the points that
           are in front of the camera only an the corresponding 3D points."""

        # 3D points in camera reference frame.
        pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))

        # Before projecting, keep only points with z > 0 
        # (points that are in fronto of the camera).
        idx = (pts3d_cam[2,:] >= 0)
        pts2d_cam = Prect.dot(pts3d_cam[:,idx])

        # return pts3d[:, idx], pts2d_cam / pts2d_cam[2,:]
        return pts2d_cam / pts2d_cam[2,:]


if __name__ == '__main__':
    # 00, 01 : gray
    # d = KITTIDataset(date='2011_09_26', driveNo='0001', color=False, sync = True)
    # print(len(d))
    # img = d[0]
    # print(img)
    # print(img.shape)
    # d = KITTIDataset(date='2011_09_26', driveNo='0001', color=False, sync = True, isLeft=False)

    # print(len(d))
    # img = d[0]
    # print(img)
    # print(img.shape)

    # 02, 03 : color
    # d = KITTIDataset(date='2011_09_26', driveNo='0001', color=True, sync = True)
    # d = KITTIDataset(date='2011_09_26', driveNo='0001', color=True, sync = True, isLeft=False)
    # local Folder
    # d = KITTIDataset(date='2011_09_26', driveNo='0005', color=True, sync = True, isLeft=False)
    d = KITTIDataset(date='2011_09_26', driveNo='0020', color=True, sync = True)
    # use pykitti
    # d = KITTIDataset(date='2011_09_26', driveNo='0001', color=True, sync = True)
    print(len(d))
    img, bbox, label = d[20]
    # print(img)
    # print(img.shape)
    # Data no Sync
    # d = KITTIDataset(date='2011_09_26', driveNo='0001', color=False, sync = False)
    # print(img.debug_print())
    vis_bbox(img, bbox, label, score=None, label_names=KITTI_label_names)
    plt.show()

