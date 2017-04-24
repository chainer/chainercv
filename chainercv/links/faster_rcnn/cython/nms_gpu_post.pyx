def nms_gpu_post(mask, boxes_num, threads_per_block):

    remv = np.zeros((col_blocks,), dtype=np.uint64)
    for i in range(boxes_num):
        nblock = i / threads_per_block
        inblock = np.array(i % threads_per_block, dtype=np.uint64)

        one_ull = np.array(1, dtype=np.uint64)
        # print np.bitwise_and(remv[nblock], np.left_shift(one_ull, inblock))
        if not np.bitwise_and(remv[nblock], np.left_shift(one_ull, inblock)):
            keep_out[num_to_keep] = i
            num_to_keep += 1

            index = i * col_blocks
            for j in range(nblock, col_blocks):
                remv[j] = np.bitwise_or(remv[j], mask_host[index + j])
    return keep_out, num_to_keep
