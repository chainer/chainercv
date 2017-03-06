def bbox_flip(bboxes, h_flip, v_flip, img_shape):
    H, W = img_shape
    if h_flip:
        x_max = W - 1 - bboxes[:, 0]
        x_min = W - 1 - bboxes[:, 2]
        bboxes[:, 0] = x_min
        bboxes[:, 2] = x_max
    if v_flip:
        y_max = H - 1 - bboxes[:, 1]
        y_min = H - 1 - bboxes[:, 3]
        bboxes[:, 1] = y_min
        bboxes[:, 3] = y_max
    return bboxes
