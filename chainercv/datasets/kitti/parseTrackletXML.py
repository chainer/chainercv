"""Parse XML files containing tracklet info for kitti data base.
   (http://cvlibs.net/datasets/kitti/raw_data.php)

   No guarantees that this code is correct, usage is at your own risk!

   created by Christian Herdtweck,
   Max Planck Institute for Biological Cybernetics
   (christian.herdtweck@tuebingen.mpg.de)

   requires numpy!

   example usage:
     import parseTrackletXML as xmlParser
     kittiDir = '/path/to/kitti/data'
     drive = '2011_09_26_drive_0001'
     xmlParser.example(kittiDir, drive)
   or simply on command line:
     python parseTrackletXML.py
"""

# Version History:
# 4/7/12 Christian Herdtweck:
#   seems to work with a few random test xml tracklet files;
#   converts file contents to ElementTree and then to list of Tracklet objects;
#   Tracklet objects have str and iter functions
# 5/7/12 ch: added constants for state, occlusion,
#            truncation and added consistency checks
# 30/1/14 ch: create example function from example code

import itertools
from warnings import warn
from xml.etree.ElementTree import ElementTree

import numpy as np

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0': STATE_UNSET, '1': STATE_INTERP, '2': STATE_LABELED}

OCC_UNSET = 255  # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1': OCC_UNSET, '0': OCC_VISIBLE,
               '1': OCC_PARTLY, '2': OCC_FULLY}

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {'99': TRUNC_UNSET, '0': TRUNC_IN_IMAGE, '1': TRUNC_TRUNCATED,
                 '2': TRUNC_OUT_IMAGE, '3': TRUNC_BEHIND_IMAGE}


class Tracklet(object):
    r"""Representation an annotated object track.

    Tracklets are created in function parseXML
      and can most conveniently used as follows:

    for trackletObj in parseXML(tracklet_filepath):
      for translation, rotation, state, occlusion, \
          truncation, amt_occlusion, amt_borders, \
          absoluteFrameNumber in trackletObj:
        ... your code here ...
      #end: for all frames
    #end: for all tracklets

    absoluteFrameNumber is in range [firstFrame, firstFrame+nFrames[
    amt_occlusion and amt_borders could be None

    You can of course also directly access the fields
    objType (string), size (len-3 ndarray), firstFrame/nFrames (int),
    trans/rots (nFrames x 3 float ndarrays),
    states/truncs (len-nFrames uint8 ndarrays),
    occs (nFrames x 2 uint8 ndarray),
    and for some tracklets amt_occs (nFrames x 2 float ndarray)
    and amt_borders (nFrames x 3 float ndarray).
    The last two can be None if the xml file
    did not include these fields in poses
    """

    objectType = None
    size = None  # len-3 float array: (height, width, length)
    firstFrame = None
    trans = None   # n x 3 float array (x,y,z)
    rots = None    # n x 3 float array (x,y,z)
    states = None  # len-n uint8 array of states
    occs = None    # n x 2 uint8 array  (occlusion, occlusion_kf)
    truncs = None  # len-n uint8 array of truncation
    # None or (n x 2) float array  (amt_occlusion, amt_occlusion_kf)
    amt_occs = None
    amt_borders = None    # None (n x 3) float array  (amt_border_l / _r / _kf)
    nFrames = None

    def __init__(self):
        """Create Tracklet with no info set."""
        self.size = np.nan*np.ones(3, dtype=float)

    def __str__(self):
        """Return human-readable string representation of tracklet object.

        called implicitly in
        # print(trackletObj)
        or in
        text = str(trackletObj)
        """
        return '[Tracklet over {0} frames for {1}]'.format(
            self.nFrames, self.objectType)

    def __iter__(self):
        """Return an iterator object.

        that yields tuple of all the available data for each frame

        called whenever code iterates over a tracklet object, e.g. in
        for translation, rotation, state, occlusion, truncation,
            amt_occlusion, amt_borders, absoluteFrameNumber in trackletObj:
          ...do something ...
        or
        trackDataIter = iter(trackletObj)
        """
        if self.amt_occs is None:
            # Python2/3
            return zip(
                self.trans, self.rots, self.states,
                self.occs, self.truncs,
                itertools.repeat(None), itertools.repeat(None),
                range(self.firstFrame, self.firstFrame+self.nFrames))
            # xrange(self.firstFrame, self.firstFrame+self.nFrames))
            # tmpAmtOccs = repeat(None)
            # tmpAmtBorders = repeat(None)
            # return zip(self.trans, self.rots, self.states,
            #     self.occs, self.truncs,
            #     tmp_amt_occs, tmp_amt_borders,
            #     range(self.firstFrame, self.firstFrame + self.nFrames))
        else:
            # Python2/3
            return zip(
                self.trans, self.rots, self.states,
                self.occs, self.truncs,
                self.amt_occs, self.amt_borders,
                range(self.firstFrame, self.firstFrame + self.nFrames))
            # xrange(self.firstFrame, self.firstFrame+self.nFrames))
            # return zip(self.trans, self.rots, self.states,
            #     self.occs, self.truncs,
            #     self.amt_occs, self.amt_borders,
            #     range(self.firstFrame, self.firstFrame + self.nFrames))
# end: class Tracklet


def parseXML(tracklet_filepath):
    r"""Parse tracklet xml file and convert list of Tracklet objects.

    :param tracklet_filepath: name of a tracklet xml file
    :returns: list of Tracklet objects read from xml file
    """
    new_track_nframes_isnone_errorstr = \
        'there are several pose lists for a single track!'

    # convert tracklet XML data to a tree structure
    element_tree = ElementTree()
    # print('parsing tracklet file', tracklet_filepath)
    with open(tracklet_filepath) as f:
        element_tree.parse(f)

    # now convert output to list of Tracklet objects
    tracklets_element = element_tree.find('tracklets')
    tracklets = []
    tracklet_idx = 0
    numeric_tracklets = None
    for tracklet_element in tracklets_element:
        # print('track:', tracklet_element.tag)
        if tracklet_element.tag == 'count':
            numeric_tracklets = int(tracklet_element.text)
            # print('file contains', numeric_tracklets, 'tracklets')
        elif tracklet_element.tag == 'item_version':
            pass
        elif tracklet_element.tag == 'item':
            # print(
            # 'tracklet {0} of {1}'.format(tracklet_idx, numeric_tracklets))
            # a tracklet
            new_track = Tracklet()
            is_finished = False
            has_amt = False
            frame_idx = None
            for info in tracklet_element:
                # print('trackInfo:', info.tag)
                if is_finished:
                    raise ValueError('more info on element after finished!')
                if info.tag == 'objectType':
                    new_track.objectType = info.text
                elif info.tag == 'h':
                    new_track.size[0] = float(info.text)
                elif info.tag == 'w':
                    new_track.size[1] = float(info.text)
                elif info.tag == 'l':
                    new_track.size[2] = float(info.text)
                elif info.tag == 'first_frame':
                    new_track.firstFrame = int(info.text)
                elif info.tag == 'poses':
                    # this info is the possibly long list of poses
                    for pose in info:
                        # print('trackInfoPose:', pose.tag)
                        # this should come before the others
                        if pose.tag == 'count':
                            if new_track.nFrames is not None:
                                raise ValueError(
                                    new_track_nframes_isnone_errorstr)
                            elif frame_idx is not None:
                                raise ValueError('?!')
                            new_track.nFrames = int(pose.text)
                            new_track.trans = np.nan * \
                                np.ones((new_track.nFrames, 3), dtype=float)
                            new_track.rots = np.nan * \
                                np.ones((new_track.nFrames, 3), dtype=float)
                            new_track.states = np.nan * \
                                np.ones(new_track.nFrames, dtype='uint8')
                            new_track.occs = np.nan * \
                                np.ones((new_track.nFrames, 2), dtype='uint8')
                            new_track.truncs = np.nan * \
                                np.ones(new_track.nFrames, dtype='uint8')
                            new_track.amt_occs = np.nan * \
                                np.ones((new_track.nFrames, 2), dtype=float)
                            new_track.amt_borders = np.nan * \
                                np.ones((new_track.nFrames, 3), dtype=float)
                            frame_idx = 0
                        elif pose.tag == 'item_version':
                            pass
                        elif pose.tag == 'item':
                            # pose in one frame
                            if frame_idx is None:
                                raise ValueError(
                                    'pose item came before number of poses!')
                            for poseInfo in pose:
                                # print('trackInfoPoseInfo:', poseInfo.tag)
                                if poseInfo.tag == 'tx':
                                    new_track.trans[frame_idx, 0] = float(
                                        poseInfo.text)
                                elif poseInfo.tag == 'ty':
                                    new_track.trans[frame_idx, 1] = float(
                                        poseInfo.text)
                                elif poseInfo.tag == 'tz':
                                    new_track.trans[frame_idx, 2] = float(
                                        poseInfo.text)
                                elif poseInfo.tag == 'rx':
                                    new_track.rots[frame_idx, 0] = float(
                                        poseInfo.text)
                                elif poseInfo.tag == 'ry':
                                    new_track.rots[frame_idx, 1] = float(
                                        poseInfo.text)
                                elif poseInfo.tag == 'rz':
                                    new_track.rots[frame_idx, 2] = float(
                                        poseInfo.text)
                                elif poseInfo.tag == 'state':
                                    new_track.states[frame_idx] = \
                                        stateFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion':
                                    new_track.occs[frame_idx, 0] = \
                                        occFromText[poseInfo.text]
                                elif poseInfo.tag == 'occlusion_kf':
                                    new_track.occs[frame_idx, 1] = \
                                        occFromText[poseInfo.text]
                                elif poseInfo.tag == 'truncation':
                                    new_track.truncs[frame_idx] = \
                                        truncFromText[poseInfo.text]
                                elif poseInfo.tag == 'amt_occlusion':
                                    new_track.amt_occs[frame_idx, 0] = \
                                        float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_occlusion_kf':
                                    new_track.amt_occs[frame_idx, 1] = \
                                        float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_l':
                                    new_track.amt_borders[frame_idx, 0] = \
                                        float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_r':
                                    new_track.amt_borders[frame_idx, 1] = \
                                        float(poseInfo.text)
                                    has_amt = True
                                elif poseInfo.tag == 'amt_border_kf':
                                    new_track.amt_borders[frame_idx, 2] = \
                                        float(poseInfo.text)
                                    has_amt = True
                                else:
                                    raise ValueError(
                                        'unexpected tag in poses item: {0}!'
                                        .format(poseInfo.tag))
                            frame_idx += 1
                        else:
                            raise ValueError(
                                'unexpected pose info: {0}!'.format(pose.tag))
                elif info.tag == 'finished':
                    is_finished = True
                else:
                    raise ValueError(
                        'unexpected tag in tracklets: {0}!'.format(info.tag))
            # end: for all fields in current tracklet

            # some final consistency checks on new tracklet
            if not is_finished:
                warn('tracklet {0} was not finished!'.format(tracklet_idx))
            if new_track.nFrames is None:
                warn('tracklet {0} contains no information!'.format(
                    tracklet_idx))
            elif frame_idx != new_track.nFrames:
                warn(
                    'tracklet {0} is supposed to have {1} frames, \
                    but perser found {1}!'.format(tracklet_idx,
                                                  new_track.nFrames,
                                                  frame_idx))
            if np.abs(new_track.rots[:, :2]).sum() > 1e-16:
                warn('track contains rotation other than yaw!')

            # if amt_occs / amt_borders are not set, set them to None
            if not has_amt:
                new_track.amt_occs = None
                new_track.amt_borders = None

            # add new tracklet to list
            tracklets.append(new_track)
            tracklet_idx += 1

        else:
            raise ValueError('unexpected tracklet info')
    # end: for tracklet list items

    # print('loaded', tracklet_idx, 'tracklets')

    # final consistency check
    if tracklet_idx != numeric_tracklets:
        warn('according to xml information the file has {0} tracklets, \
            but parser found {1}!'.format(numeric_tracklets, tracklet_idx))

    return tracklets
# end: function parseXML
