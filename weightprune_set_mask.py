from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import *
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse
import torch.distributed as dist
from utils.datasets import *
from torch.utils.tensorboard import SummaryWriter


# Hyper Parameters
param = {
    'pruning_perc': 90.,
    'batch_size': 20,
    'test_batch_size': 20,
    'num_epochs': 2,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
}
# Hyperparameters (j-series, 50.5 mAP yolov3-320) evolved by @ktian08 https://github.com/ultralytics/yolov3/issues/310
hyp = {'giou': 1.582,  # giou loss gain
       'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
       'cls_pw': 1.446,  # cls BCELoss positive_weight
       'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
       'obj_pw': 3.941,  # obj BCELoss positive_weight
       'iou_t': 0.2635,  # iou training threshold
       'lr0': 0.002324,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.97,  # SGD momentum
       'weight_decay': 0.0004569,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.10,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
       'degrees': 1.113,  # image rotation (+/- deg)
       'translate': 0.06797,  # image translation (+/- fraction)
       'scale': 0.1059,  # image scale (+/- gain)
       'shear': 0.5768}  # image shear (+/- deg)
wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-hand.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/oxfordhand.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.8, help='channel prune percent')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--group_num', type=int, default=8, help='layer weights group num')
    parser.add_argument('--cut_num', type=int, default=2, help='weights cut num')
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--arc', type=str, default='defaultpw', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    opt = parser.parse_args()
    print(opt)


    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)

    if opt.weights.endswith('.pt'):
        model.load_state_dict(torch.load(opt.weights, map_location=torch.device('cpu'))['model'])
    else:
        load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ',opt.weights)
    print("--- Pretrained network loaded ---")

    check0 = model.module_list[1][0].weight.data.cpu().numpy()[0, :, 0, 0]
    check1 = model.module_list[25][0].weight.data.cpu().numpy()[0, :, 0, 0]
    check2 = model.module_list[112][0].weight.data.cpu().numpy()[0, :, 0, 0]
    print(check0, "\n********************\n", check1, "\n*****************\n", check2)

    eval_model = lambda model:test(opt.cfg, opt.data,
        batch_size=16,
         img_size=img_size,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False,
         model=model)
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])
    origin_nparameters = obtain_num_parameters(model)  # 获取原始参数量

    print("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model)

    def obtain_avg_forward_time(input, model, repeat=200):

        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)[0]
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output
    print('\ntesting avg forward time...')
    random_input = torch.rand((1, 3, img_size, img_size)).to(device)
    origin_forward_time, origin_output = obtain_avg_forward_time(random_input, model)


    '''def store_new_mask(init_list, n):
        new_mask = []
        for i in range(0, len(init_list), n):
            child_mask = init_list[i:i+n]
            new_mask.append(child_mask)
        return new_mask'''

    def weight_prune(module_list, Conv_together, group_num, cut_num):
        size_list = []
        pruned_idx = []
        masks_list = []
        ori_param_num = 0
        pruned_param_num = 0
        for idx in Conv_together:
            size = module_list[idx][0].weight.data.shape[1]
            if size > group_num:
                weight = module_list[idx][0].weight.data.cpu().numpy()
                conv_shape = weight.shape
                k_num = weight.shape[0]
                c = weight.shape[1]
                h = weight.shape[2]
                w = weight.shape[3]
                #size_list.append(size)
                l_param = k_num * c * h * w
                ori_param_num += l_param
                pruned_idx.append(idx)

                # conv_weights = torch.zeros(sum(size_list))
                devided_num = int(size) // group_num

                masks = np.ones((conv_shape))

                for k in range(k_num):
                    for h_idx in range(h):
                        for w_idx in range(w):
                            for i in range(int(devided_num)):
                                c_weights = np.abs(weight[k, i*group_num:(i+1)*group_num, h_idx, w_idx])
                                zero_count = np.sum(c_weights == 0)

                                weights_copy = deepcopy(c_weights)
                                weights_copy.sort()
                                if zero_count == 0:
                                    threshold = weights_copy[cut_num - 1]
                                    pruned_c = np.array(c_weights > threshold).astype(int)
                                    masks[k, i*group_num:(i+1)*group_num, h_idx, w_idx] = pruned_c
                                else:
                                    threshold = weights_copy[zero_count + cut_num - 1]
                                    pruned_c = np.array(c_weights > threshold).astype(int)
                                    masks[k, i * group_num:(i + 1) * group_num, h_idx, w_idx] = pruned_c


                masks_list.append(masks)
                l_param_remain = k_num * h * w * (c - devided_num * (cut_num + zero_count))
                pruned_param_num += l_param_remain
                # conv_weights = weight.cpu().abs().detach().numpy().transpose((0, 2, 3, 1)).flatten()
                # conv_weights = module_list[idx][0].weight.T.cpu().abs().detach().numpy().flatten()
                #masks_list.append(c_weights)
        print("now zero counts are:", zero_count)
        print('cut all {} conv layers'.format(len(pruned_idx)))
        print('---masks_list already get!---')
        print('count origin params:{}\t count pruned params:{}'.format(ori_param_num, pruned_param_num))
        return pruned_idx, masks_list



    group_num = opt.group_num
    cut_num = opt.cut_num
    pruning_perc = cut_num/group_num
    print('the required prune percent is : ', pruning_perc)


    # 提取需要裁剪的层的id
    CBL_idx, Conv_idx, Conv_together = parse_module_defs_userconv(model.module_defs)

    #print(model.module_list[112][0].weight.data.cpu().numpy()[17, :, 0, 0])
    # 提取需要裁剪的层的BN参数
    #conv_weights_list, prune_idx = gather_conv_weights0(model.module_list, Conv_together, group_num)
    prune_idx, masks_list = weight_prune(model.module_list, Conv_together, group_num, cut_num)

    Convidx2mask = {idx: mask.astype('float32') for idx, mask in zip(prune_idx, masks_list)}

    model.set_masks(prune_idx, Convidx2mask)
    print(model.module_list[1][0].weight.data.cpu().numpy()[0,:,0,0])

    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    print('testing the mAP of final pruned model')
    with torch.no_grad():
        compact_model_metric = eval_model(model)

    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, model)
    compact_nparameters = obtain_num_parameters(model)
    #np.array(conv_weights_list).mul_(np.array(masks_list))

    #prune_weights = map(lambda a_b: a_b[0] * a_b[1], zip(conv_weights_list, masks_list))
    #prune_weights = np.array(list(prune_weights))



    # Retraining
    def train_user0(model):
        data = opt.data
        cfg = opt.cfg
        img_size = opt.img_size
        batch_size = opt.batch_size
        epochs = opt.epochs
        accumulate = opt.accumulate

        if 'pw' not in opt.arc:  # remove BCELoss positive weights
            hyp['cls_pw'] = 1.
            hyp['obj_pw'] = 1.

        # Initialize
        init_seeds()

        # Configure run 获取数据路径
        data_dict = parse_data_cfg(data)
        train_path = data_dict['train']
        nc = int(data_dict['classes'])# number of classes
        start_epoch = 0
        best_fitness = 0.
        # Dataset
        dataset = LoadImagesAndLabels(train_path,
                                      img_size,
                                      batch_size,
                                      augment=True,
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=opt.rect,  # rectangular training
                                      cache_labels=True if epochs > 10 else False,
                                      cache_images=opt.cache_images)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=min([os.cpu_count(), batch_size, 16]),
                                                 shuffle=not opt.rect,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)

        # Remove previous results 移除之前的结果
        for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
            os.remove(f)


        def adjust_learning_rate(optimizer, gamma, epoch, iteration, epoch_size):
            """调整学习率进行warm up和学习率衰减
            """
            step_index = 0
            if epoch < 6:
                # 对开始的6个epoch进行warm up
                lr = 1e-6 + (hyp['lr0'] - 1e-6) * iteration / (epoch_size * 2)
            else:
                if epoch > opt.epochs * 0.7:
                    # 在进行总epochs的70%时，进行以gamma的学习率衰减
                    step_index = 1
                if epoch > opt.epochs * 0.9:
                    # 在进行总epochs的90%时，进行以gamma^2的学习率衰减
                    step_index = 2

                lr = hyp['lr0'] * (gamma ** (step_index))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return lr


        #criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=param['learning_rate'],
                                        #weight_decay=param['weight_decay'])
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'],
                              weight_decay=hyp['weight_decay'], nesterov=True)
        # Start training
        model.nc = nc  # attach number of classes to model
        model.arc = opt.arc  # attach yolo architecture
        model.hyp = hyp  # attach hyperparameters to model

        torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
        nb = len(dataloader)
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'

        t0 = time.time()
        print('Starting %s for %g epochs...' % ('training', epochs))
        writer = SummaryWriter('yolov3')
        for epoch in range(start_epoch,epochs):  # epoch ------------------------------------------------------------------
            model.train()
            # print('learning rate:',optimizer.param_groups[0]['lr'])
            print(('\n' + '%10s' * 8) % (
            'Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))


            mloss = torch.zeros(4).to(device)  # mean losses
            pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
            for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)

                # 调整学习率，进行warm up和学习率衰减
                lr = adjust_learning_rate(optimizer, 0.1, epoch, ni, nb)
                if i == 0:
                    print('\nlearning rate:', lr)

                imgs = imgs.to(device)
                targets = targets.to(device)

                # Plot images with bounding boxes
                if ni == 0:
                    fname = 'train_batch%g.jpg' % i
                    plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)

                # Run model
                pred = model(imgs)

                # Compute loss
                loss, loss_items = compute_loss(pred, targets, model)
                if (i+1) % 20 == 0:
                    print('i = %d, loss = %.8f' % (i, loss.item()))
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                # Scale loss by nominal batch_size of 64
                #loss *= batch_size / 64

                # Compute gradient
                loss.backward()

                # Accumulate gradient for x batches before optimizing
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()


                # Print batch results
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
                pbar.set_description(s)
                # end batch ------------------------------------------------------------------------------------------------

            # Process epoch results
            final_epoch = epoch + 1 == epochs

            # Calculate mAP (always test final epoch, skip first 10 if opt.nosave)
            #if (epoch+1) % 2 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                results, maps = test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      img_size=opt.img_size,
                                      model=model,
                                      conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,
                                      # 0.1 for speed
                                      save_json=final_epoch and epoch > 0 and 'coco.data' in data)

            # Write epoch results
            with open(results_file, 'a') as f:
                f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)


            # Update best mAP
            fitness = results[2]  # mAP
            if fitness > best_fitness:
                best_fitness = fitness
            writer.add_scalar("Train/Loss", loss.item(), i)
            writer.add_scalar("Train/Accuracy", fitness, epoch)

            # Save training results
            '''save = (epoch == epochs - 1)
            if save:

                with open(results_file, 'r') as f:
                    # Create checkpoint
                    chkpt = {'epoch': epoch,
                             'best_fitness': best_fitness,
                             'training_results': f.read(),
                             'model': model.module.state_dict() if type(
                                 model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                             'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last checkpoint
                torch.save(chkpt, last)

                # Save best checkpoint
                if best_fitness == fitness:
                    torch.save(chkpt, best)

                # Delete checkpoint
                del chkpt'''
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training
        if len(opt.name):
            os.rename('results.txt', 'results_%s.txt' % opt.name)
        #plot_results()  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        #dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
        return results



    train_user0(model)
    print("--- After retraining ---")
    check6 = model.module_list[1][0].weight.data.cpu().numpy()[0, :, 0, 0]
    check7 = model.module_list[25][0].weight.data.cpu().numpy()[0, :, 0, 0]
    check8 = model.module_list[112][0].weight.data.cpu().numpy()[0, :, 0, 0]
    print(check6,"\n********************\n",check7,"\n*****************\n",check8)
    print("check the model weather pruned!",np.size(check6),np.size(check7),np.size(check8))
    with torch.no_grad():
        retrained_model_metric = eval_model(model)
    # %%
    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", {origin_model_metric[0][2]}, {compact_model_metric[0][2]}, {retrained_model_metric[0][2]}],
        ["Parameters", {origin_nparameters}, {compact_nparameters}],
        ["Inference", {origin_forward_time}, {pruned_forward_time}]
    ]
    print(AsciiTable(metric_table).table)


    # %%
    # 生成剪枝后的cfg文件并保存模型
    # pruned_cfg_name = opt.cfg.replace('/', '/prune_{}_{}'.format(group_num, cut_num))
    # pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    # print('Config file has been saved:', pruned_cfg_file)

    '''compact_model_name = opt.weights.replace('/', '/prune_{}_{}_setmask0_'.format(group_num, cut_num))
    if compact_model_name.endswith('.pt'):
        compact_model_name = compact_model_name.replace('.pt', '.weights')
    save_weights(model, compact_model_name)
    print('Compact model has been saved: ', compact_model_name)'''