import argparse
import logging
import os
import sys
import time

from tensorboardX import SummaryWriter
from tqdm import tqdm

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils import ramps, losses, metrics, test_patch


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default= '', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='SCMT', help='exp_name')
parser.add_argument('--model', type=str, default='SCMT3d', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
#parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--num_classes', type=int, default=2, help='weight to balance all losses')
parser.add_argument('--ema_decay', type=float, default=0.99, help='weight to balance all losses')
args = parser.parse_args()

snapshot_path = "{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp,args.labelnum, args.model)

if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = ''
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = ''
    args.max_samples = 62
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha,param.data)



if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)


    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        """Network definition"""
        net = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes, mode="train")
        model = net.cuda()
        if ema:
          for param in model.parameters():
              param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)


    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    num_classes = args.num_classes
    labelnum = args.labelnum
    labeled_idxs = list(range())
    unlabeled_idxs = list(range())
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - labeled_bs)
    print("有标签索引列表：{},无标签索引列表：{}".format(labeled_idxs, unlabeled_idxs))



    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)



    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)
    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    iterator = tqdm(range(max_epoch), ncols=70)
    print()


    model.train()
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
           #img_s = sampled_batch['img_s'].cuda()   #强增强图片


            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            unlboutputs=model(unlabeled_volume_batch)
            num_outputs = len(outputs)

            with torch.no_grad():
                ema_outputs = ema_model(ema_inputs)

            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():

                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)[2]
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, 112, 112, 80)
            preds = torch.mean(preds, dim=0)


            y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
            num_ema_outputs = len(ema_outputs)
            y_ema_ori =torch.zeros((num_ema_outputs,) + ema_outputs[0].shape)
            num_unlboutputs = len(unlboutputs)
            y_unlb = torch.zeros((num_unlboutputs,) + unlboutputs[0].shape)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)

            loss_seg_ce = 0
            loss_seg_dice = 0
            #r_drop_loss=0
            for idx in range(num_outputs):
                y_lb = outputs[idx][:labeled_bs, ...]

                y_prob = F.softmax(y_lb, dim=1)
                ## calculate the supervised_loss
                loss_seg_ce += F.cross_entropy(y_lb[:labeled_bs], label_batch[:labeled_bs])
                loss_seg_dice += dice_loss(y_prob[:, 1, ...], label_batch[:labeled_bs,...] == 1)
                supervised_loss = loss_seg_ce+ loss_seg_dice


                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all

                ema_outputs_all=ema_outputs[idx]
                y_ema_ori[idx]=F.softmax(ema_outputs_all,dim=1)

                y_unlb_all=unlboutputs[idx]
                y_unlb[idx]=F.softmax(y_unlb_all,dim=1)

                y_pseudo_label[idx] = sharpening(y_prob_all)
                max_probs, max_idx = torch.max(y_pseudo_label, dim=-1)

            ## calculate the consist loss and total loss
            loss_consist = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
            consistency_weight = get_current_consistency_weight(iter_num // 150)  #
            consistency_dist = consistency_criterion(y_unlb, y_ema_ori)


            consistency_loss = consistency_weight * (consistency_dist)

            r_drop_loss = losses.compute_kl_loss(y_unlb, y_ema_ori)
            consistency_loss = consistency_weight * r_drop_loss
            total_loss = args.lamda * supervised_loss+consistency_loss*(1-args.lamda)



            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1


            writer.add_scalar('loss/loss', total_loss, iter_num)
            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg_ce, iter_num)
            writer.add_scalar('Labeled_loss/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('Co_loss/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', consistency_loss, iter_num)


            logging.info('iteration %d : loss : %03f, loss_d: %03f,loss_ce: %03f, supervised_loss: %03f,floss_cosist: %03f,loss_weight: %03f,consistency_loss: %03f'
                         %(iter_num, total_loss, loss_seg_dice, loss_seg_ce, supervised_loss,loss_consist,consistency_weight,consistency_loss))  


            if iter_num >= 800 and iter_num % 200 == 0:
                ins_width = 2
                B, C, H, W, D = y_ori[0].size()
                snapshot_img = torch.zeros(
                    size=(D, 3, (num_outputs + 2) * H + (num_outputs + 2) * ins_width, W + ins_width),
                    dtype=torch.float32)


                target = label_batch[labeled_bs, ...].permute(2, 0, 1)
                train_img = volume_batch[labeled_bs, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, :, :, W:W + ins_width] = 1
                for idx in range(num_outputs + 2):
                    begin_grid = idx + 1
                    snapshot_img[:, :, begin_grid * H + ins_width:begin_grid * H + begin_grid * ins_width,:] = 1


                for idx in range(num_outputs):
                    begin = idx + 2
                    end = idx + 3
                    snapshot_img[:, 0, begin * H + begin * ins_width:end * H + begin * ins_width, :W] = \
                    y_ori[idx][labeled_bs:][0, 1].permute(2, 0, 1)
                    snapshot_img[:, 1, begin * H + begin * ins_width:end * H + begin * ins_width, :W] = \
                    y_ori[idx][labeled_bs:][0, 1].permute(2, 0, 1)
                    snapshot_img[:, 2, begin * H + begin * ins_width:end * H + begin * ins_width, :W] = \
                    y_ori[idx][labeled_bs:][0, 1].permute(2, 0, 1)
                writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch_num, iter_num),
                                  snapshot_img)

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()




                if args.dataset_name == "LA":
                    dice_sample =test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                                stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample =test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                                stride_xy=16, stride_z=16, dataset_name='Pancreas_CT')
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num,best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample,iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice,iter_num)

                model.train()
            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            iterator.close()  
            break
    writer.close()
