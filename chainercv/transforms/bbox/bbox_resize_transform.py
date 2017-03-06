def bbox_resize(bboxes, input_shape, output_shape):
    bboxes = bboxes.copy()
    h_scale = float(output_shape[1]) / input_shape[0]
    v_scale = float(output_shape[0]) / input_shape[0]
    bboxes[:, 0] = h_scale * bboxes[:, 0]
    bboxes[:, 2] = h_scale * bboxes[:, 2]
    bboxes[:, 1] = v_scale * bboxes[:, 1]
    bboxes[:, 3] = v_scale * bboxes[:, 3]
    return bboxes
