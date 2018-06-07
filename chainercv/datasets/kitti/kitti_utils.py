# tracklet parser
from chainercv.datasets.kitti import parseTrackletXML as xmlParser

import numpy as np
import os
from urllib.parse import urljoin

from chainercv import utils

from chainer.dataset import download

# root = 'pfnet/chainercv/kitti'
url_base = 'http://kitti.is.tue.mpg.de/kitti/raw_data/'


def get_kitti_sync_data(root, date, driveNo):
    data_root = download.get_dataset_directory(root)

    # data
    folder = date + '_drive_' + driveNo
    url_data = urljoin(url_base, folder + '/' + folder + '_sync.zip')

    # calibration
    url_calib = url_base + date + '_calib.zip'

    # tracklet
    url_tracklet = urljoin(url_base, folder + '/' + folder + '_tracklets.zip')

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


def get_kitti_nosync_data(root, date, driveNo):
    data_root = download.get_dataset_directory(root)

    # data
    folder = date + '_drive_' + driveNo
    url_data = urljoin(url_base, folder + '/' + folder + '_extract.zip')

    # calibration
    url_calib = url_base + date + '_calib.zip'

    # tracklet
    url_tracklet = urljoin(url_base, folder + '/' + folder + '_tracklets.zip')

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


def get_kitti_tracklets(data_root, date, driveNo):
    # read calibration files
    kitti_dir = os.path.join(data_root, date)
    # kitti_dir = kitti_dir.replace(os.path.sep, '/')
    # calibration_dir = os.path.join(data_root, date)
    # imu2velo = read_calib_file(
    #                os.path.join(kitti_dir, "calib_imu_to_velo.txt"))
    # velo2cam = read_calib_file(
    #                os.path.join(kitti_dir, "calib_velo_to_cam.txt"))
    # cam2cam = read_calib_file(
    #                os.path.join(kitti_dir, "calib_cam_to_cam.txt"))
    # read tracklet
    folder = date + '_drive_' + driveNo + '_sync'
    # tracklet = read_tracklet_file(
    #                os.path.join(kitti_dir, folder, "calib_imu_to_velo.txt"))
    # return tracklets
    # get dir names
    # read tracklets from file
    myTrackletFile = os.path.join(kitti_dir, folder, 'tracklet_labels.xml')
    tracklets = xmlParser.parseXML(myTrackletFile)
    return tracklets


def get_kitti_label(tracklets, calib, cur_R_rect, cur_P_rect, framelength):
    # set list
    bboxes = [0] * framelength
    labels = [0] * framelength
    for idx in range(0, framelength):
        bboxes[idx] = []
        labels[idx] = []

    # set ndarray
    # bboxes = np.zeros(framelength, dtype=np.float32)
    # labels = np.zeros(framelength, dtype=np.int32)

    # twoPi = 2.*np.pi
    # loop over tracklets
    for iTracklet, tracklet in enumerate(tracklets):
        # print('tracklet {0: 3d}: {1}'.format(iTracklet, tracklet))

        # this part is inspired by kitti object development kit
        # matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2],
            [w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2],
            [0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

        # print('trackletBox : ' + trackletBox)
        # print(trackletBox)
        objTypeStr = tracklet.objectType
        # print(objTypeStr)

        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, \
                amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:

            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE,
                                  xmlParser.TRUNC_TRUNCATED):
                continue

            # re-create 3D bounding box in velodyne coordinate system
            # other rotations are 0 in all xml files I checked
            yaw = rotation[2]
            assert np.abs(rotation[:2]).sum(
            ) == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0, 1.0]])
            cornerPosInVelo = np.dot(
                rotMat, trackletBox) + np.tile(translation, (8, 1)).T

            # calc yaw as seen from the camera
            # (i.e. 0 degree = facing away from cam),
            # as opposed to car-centered yaw
            # (i.e. 0 degree = same orientation as car).
            # makes quite a difference for objects in periphery!
            # Result is in [0, 2pi]
            x, y, z = translation
            # print(translation)
            # yawVisual = ( yaw - np.arctan2(y, x) ) % twoPi
            # print(yaw)
            # print(yawVisual)
            # param = pykitti.utils.transform_from_rot_trans(
            #             rotMat, translation)
            # print(param)

            # projection to image?
            # print(calib.P_rect_20)
            # param3 = translation.reshape(3, 1) * calib.P_rect_20
            # print(cornerPosInVelo[:, 0:1].shape)
            pt3d = np.vstack((cornerPosInVelo[:, 0:8], np.ones(8)))
            # print(pt3d.shape)
            # print(calib.P_rect_20)
            pt2d = project_velo_points_in_img(
                pt3d, calib.T_cam2_velo, cur_R_rect, cur_P_rect)

            # print(pt2d)
            xmin = min(pt2d[0, :])
            xmax = max(pt2d[0, :])
            ymin = min(pt2d[1, :])
            ymax = max(pt2d[1, :])
            if xmin < 0.0:
                xmin = 0.0
            if ymin < 0.0:
                ymin = 0.0
            if xmax < 0.0:
                xmax = 0.0
            if ymax < 0.0:
                ymax = 0.0

            # img_size_x = img_size[0]
            # img_size_y = img_size[1]
            # image_shape = 375, 1242
            if xmin > 1242.0:
                xmin = 1242.0
            if ymin > 375.0:
                ymin = 375.0
            if xmax > 1242.0:
                xmax = 1242.0
            if ymax > 375.0:
                ymax = 375.0

            param = np.array((ymin, xmin, ymax, xmax), dtype=np.float32)
            # print(param)
            # bbox.append(param)
            # bbox = np.stack(bbox).astype(np.float32)
            # bboxes[absoluteFrameNumber] = bbox
            bboxes[absoluteFrameNumber].append(param)
            # print(bboxes[absoluteFrameNumber])

            # param_3d = cornerPosInVelo
            # bboxes_3d[absoluteFrameNumber].append(cornerPosInVelo)
            # label.append(param2)
            # label = np.stack(label).astype(np.int32)
            # labels[absoluteFrameNumber] = label
            # objectType
            # label_names
            # not search objTypeStr? process
            param2 = kitti_bbox_label_names.index(objTypeStr)
            labels[absoluteFrameNumber].append(param2)
            # labels[absoluteFrameNumber] = param2
            # print(bboxes[absoluteFrameNumber])

        # end : for all frames in track
    # end : for all tracks

    return bboxes, labels


def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    """
    Project 3D points into 2D image. Expects pts3d as a 4xN
    numpy array. Returns the 2D projection of the points that
    are in front of the camera only an the corresponding 3D points.
    """

    # 3D points in camera reference frame.
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))

    # Before projecting, keep only points with z > 0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2, :] >= 0)
    pts2d_cam = Prect.dot(pts3d_cam[:, idx])

    # return pts3d[:, idx], pts2d_cam / pts2d_cam[2,:]
    return pts2d_cam / pts2d_cam[2, :]


# image_shape = 375, 1242
# kitti_category_names = (
#     'City',
#     'Residential',
#     'Road',
#     'Campus',
#     'Person',
#     'Calibration'
# )

kitti_bbox_label_names = (
    'Car',
    'Van',
    'Truck',
    'Pedestrian',
    'Sitter',
    'Cyclist',
    'Tram',
    'Misc',
)

kitti_bbox_label_colors = (
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (60, 40, 222),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
)
kitti_ignore_bbox_label_color = (0, 0, 0)
