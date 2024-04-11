# torch and visulization
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.some_nets import  unet_resnet_18


import os


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter = load_param(args.channel_size)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir: object = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            # model =  dna_net_18()
            model = unet_resnet_18()
            # model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_Block, nb_filter=nb_filter)
            # model = acm_res_unet(num_classes=1, input_channels=args.in_channels, block=Res_Block, nb_filter=nb_filter)

        model           = model
        model.apply(weights_init_xavier)
        print("Model Initializing")
        print("Base")

        self.model      = model.cuda()

        devices = []
        sd = self.model.state_dict()
        for v in sd.values():
            if v.device not in devices:
                devices.append(v.device)
        for d in devices:
            print(d)

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)


        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data,data_1, labels) in enumerate(tbar):
            data   = data.cuda()
            data_1 = data_1.cuda()
            labels = labels.cuda()

            pred = self.model(data, data_1)
            loss = SoftIoULoss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.scheduler.step()
        self.train_loss = losses.avg

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                pred = self.model(data,data)
                # print(pred.size())
                loss = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))
            test_loss=losses.avg
        # save high-performance model
        save_model(mean_IOU, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch)

        # 保存训练的最优模型
        if mean_IOU > self.best_iou:
            save_mIoU_dir = 'result/' + self.save_dir + '/' + self.save_prefix + '_best_IoU_IoU.log'
            save_other_metric_dir = 'result/' + self.save_dir + '/' + self.save_prefix + '_best_IoU_other_metric.log'
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            self.best_iou = mean_IOU
            save_model_and_result(dt_string, epoch, self.train_loss, test_loss, self.best_iou,
                                  recall, precision, save_mIoU_dir, save_other_metric_dir)
            save_ckpt({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'loss': test_loss,
                'mean_IOU': mean_IOU,
            }, save_path='result/' + self.save_dir,
                filename='mIoU_' + '_' + self.save_prefix + '_epoch' + '.pth.tar')


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    # with torch.cuda.device(3):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    args = parse_args()
    main(args)




