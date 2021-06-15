from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-hand.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/oxfordhand.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/xishu.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.8, help='channel prune percent')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--group_num', type=int, default=8, help='layer weights group num')
    parser.add_argument('--cut_num', type=int, default=2, help='weights cut num')
    opt = parser.parse_args()
    print(opt)


    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)
    model.eval()
    if opt.weights.endswith('.pt'):
        model.load_state_dict(torch.load(opt.weights, map_location=torch.device('cpu'))['model'])
    else:
        load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ',opt.weights)

    eval_model = lambda model:test(opt.cfg, opt.data,
        weights=opt.weights,
        batch_size=16,
         img_size=img_size,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False,
         model=model)
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    print("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model)


    '''def weight_prune(idx, model, group_num, cut_num):
        layer_weights = []
        masks = []
        for p in model.parameters():

            if p.data.shape[0] > group_num:
                layer_weights = list(p.cpu().T.abs().detach().numpy().flatten())
                devided_num = int(len(layer_weights))/group_num
                index = 0
                for i in range(int(devided_num)):
                    devided_layer_weights = layer_weights[index:(index + group_num)]
                    index += group_num
                    devided_layer_weights.sort()
                    threshold = devided_layer_weights[cut_num-1]
                    pruned_inds = np.array(devided_layer_weights > threshold).astype(int)
                    masks.append(pruned_inds)
        #threshold = np.percentile(np.array(all_weights), pruning_perc)
        # generate mask

        return masks'''

    def store_new_mask(init_list, n):
        new_mask = []
        for i in range(0, len(init_list), n):
            child_mask = init_list[i:i+n]
            new_mask.append(child_mask)
        return new_mask

    def weight_prune(conv_weights_list, group_num, cut_num):
        '''
        Prune pruning_perc% weights globally (not layer-wise)
        arXiv: 1606.09274
        '''
        masks_list = []

        for idx in range(len(conv_weights_list)):
            devided_num = int(len(conv_weights_list[idx]))/group_num
            index = 0
            masks = []

            for i in range(int(devided_num)):
                devided_layer_weights = conv_weights_list[idx][index:(index + group_num)]
                index += group_num
                weights_copy = deepcopy(devided_layer_weights)
                weights_copy.sort()
                threshold = weights_copy[cut_num-1]
                #print('weight value that less than %.4f are set to zero!' % threshold)
                pruned_inds = np.array(devided_layer_weights > threshold).astype(int)
                masks.append(pruned_inds)


            masks = np.array(masks).flatten()
            masks_list.append(masks)
            #remain = int(masks.sum())

        return masks_list

    origin_nparameters = obtain_num_parameters(model)  # 获取原始参数量
    group_num = opt.group_num
    cut_num = opt.cut_num
    pruning_perc = cut_num/group_num
    print('the required prune percent is', pruning_perc)





    # 提取需要裁剪的层的id
    CBL_idx, Conv_idx, prune_idx = parse_module_defs_user(model.module_defs)

    # 提取需要裁剪的层的BN参数
    conv_weights_list = gather_conv_weights0(model.module_list, prune_idx, group_num)
    #conv_weights = gather_conv_weights(model.module_list, prune_idx)
    masks_list = weight_prune(conv_weights_list, group_num, cut_num)
    Convidx2mask = {idx: mask.astype('float32') for idx, mask in zip(prune_idx, masks_list)}
    np.array(conv_weights_list).mul_(np.array(masks_list))


    #prune_weights = map(lambda a_b: a_b[0] * a_b[1], zip(conv_weights_list, masks_list))
    #prune_weights = np.array(list(prune_weights))

    new_conv_list = []
    for i, idx in zip(range(len(conv_weights_list)), prune_idx):
        new_conv = np.array(conv_weights_list[i]).reshape(model.module_list[idx][0].weight.T.shape)
        new_conv_list.append(new_conv)



    #%%
    '''def obtain_filters_mask(model, thre, CBL_idx, prune_idx):

        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            # 如果该层需要裁剪，则先确定裁剪后的最小通道数min_channel_num，然后根据裁剪阈值进行通道裁剪确定mask，如果整层的通道γ均低于阈值，为了避免整层被裁剪，留该层中γ值最大的几个（根据layer_keep参数进行设置，最小为1）通道。
            if idx in prune_idx:

                mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                if remain == 0:
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = bn_module.weight.data.abs().max()
                    mask = obtain_bn_mask(bn_module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain

                    #print('layer index: %.3d \t total channel: %.4d \t remaining channel: %.4d' % (
                    idx, mask.shape[0], remain))
            # 如果该层不需要裁剪，则全部保留
            else:
                mask = np.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())

        prune_ratio = pruned / total
        print('Prune channels: %.3f' % pruned, '\t', 'Prune ratio: %.3f' % prune_ratio)

        return num_filters, filters_mask'''

    #num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)

    #%%
    #CBLidx2mask存储CBL_idx中，每一层BN层对应的mask


    #pruned_model = prune_model_keep_size2_user(model, prune_idx, Convidx2mask)
    '''pruned_model = deepcopy(model)
    for i in prune_idx:
        mask = torch.from_numpy(Convidx2mask[i]).cpu()
        conv_module = pruned_model.module_list[i][0]
        conv_module.weight.T.mul_(mask)
        #torch.FloatTensor(conv_module).mul_(mask)'''



    print("\nnow prune the model but keep size, let's see how the mAP goes")

    compact_module_defs = deepcopy(model.module_defs)
    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(pruned_model)
    random_input = torch.rand((1, 3, img_size, img_size)).to(device)


    def obtain_avg_forward_time(input, model, repeat=200):

        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)[0]
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output


    print('\ntesting avg forward time...')
    origin_forward_time, origin_output = obtain_avg_forward_time(random_input, model)
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)

    # %%
    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    print('testing the mAP of final pruned model')
    with torch.no_grad():
        compact_model_metric = eval_model(pruned_model)

     # %%
    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", {origin_model_metric[0][2]}, {compact_model_metric[0][2]}],
        ["Parameters", {origin_nparameters}, {compact_nparameters}],
        ["Inference", {origin_forward_time}, {pruned_forward_time}]
    ]
    print(AsciiTable(metric_table).table)

    #%%
    # 获得原始模型的module_defs，并修改该defs中的卷积核数量
    compact_module_defs = deepcopy(model.module_defs)
    for idx, num in zip(prune_idx, num_filters):
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(num)

    #%%
    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)

    init_weights_from_loose_model_user(compact_model, pruned_model, prune_idx, Convidx2mask)

    #%%

    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    diff = (pruned_output-compact_output).abs().gt(0.001).sum().item()
    if diff > 0:
        print('Something wrong with the pruned model!')

    #%%
    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    print('testing the mAP of final pruned model')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)


    #%%
    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", {origin_model_metric[0][2]}, {compact_model_metric[0][2]}],
        ["Parameters", {origin_nparameters}, {compact_nparameters}],
        ["Inference", {pruned_forward_time}, {compact_forward_time}]
    ]
    print(AsciiTable(metric_table).table)

    #%%
    # 生成剪枝后的cfg文件并保存模型
    pruned_cfg_name = opt.cfg.replace('/', '/prune_{}_'.format(percent))
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print('Config file has been saved:', pruned_cfg_file)

    compact_model_name = opt.weights.replace('/', '/prune_{}_'.format(percent))
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(compact_model, compact_model_name)
    print('Compact model has been saved: ', compact_model_name)
