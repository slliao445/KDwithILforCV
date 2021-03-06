from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

import argparse
import os
import time

from util import AverageMeter, accuracy, transform_time
from util import load_pretrained_model, save_checkpoint
from network import define_tsnet

parser = argparse.ArgumentParser(description='flow of solution procedure (FSP)')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--s_init', type=str, required=True, help='initial parameters of student model')
parser.add_argument('--t_model', type=str, required=True, help='path name of teacher model')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset')# cifar10/cifar100
parser.add_argument('--t_name', type=str, required=True, help='name of teacher')
parser.add_argument('--s_name', type=str, required=True, help='name of student')

# hyperparameter lambda
parser.add_argument('--lambda_fsp', type=float, default=1.0)

def main():
	global args
	args = parser.parse_args()
	print(args)

	if not os.path.exists(os.path.join(args.save_root,'checkpoint')):
		os.makedirs(os.path.join(args.save_root,'checkpoint'))

	if args.cuda:
		cudnn.benchmark = True

	print('----------- Network Initialization --------------')
	snet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.s_init)
	load_pretrained_model(snet, checkpoint['net'])

	tnet = define_tsnet(name=args.t_name, num_class=args.num_class, cuda=args.cuda)
	checkpoint = torch.load(args.t_model)
	load_pretrained_model(tnet, checkpoint['net'])
	tnet.eval()
	for param in tnet.parameters():
		param.requires_grad = False
	print('-----------------------------------------------')

	# initialize optimizer
	optimizer = torch.optim.SGD(snet.parameters(),
								lr = args.lr,
								momentum = args.momentum,
								weight_decay = args.weight_decay,
								nesterov = True)

	# define loss functions
	if args.cuda:
		criterionCls = torch.nn.CrossEntropyLoss().cuda()
		criterionFSP = torch.nn.MSELoss().cuda()
	else:
		criterionCls = torch.nn.CrossEntropyLoss()
		criterionFSP = torch.nn.MSELoss()

	# define transforms
	if args.data_name == 'cifar10':
		dataset = dst.CIFAR10
		mean = (0.4914, 0.4822, 0.4465)
		std  = (0.2470, 0.2435, 0.2616)
	elif args.data_name == 'cifar100':
		dataset = dst.CIFAR100
		mean = (0.5071, 0.4865, 0.4409)
		std  = (0.2673, 0.2564, 0.2762)
	else:
		raise Exception('invalid dataset name...')

	train_transform = transforms.Compose([
			transforms.Pad(4, padding_mode='reflect'),
			transforms.RandomCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])
	test_transform = transforms.Compose([
			transforms.CenterCrop(32),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean,std=std)
		])

	# define data loader
	train_loader = torch.utils.data.DataLoader(
			dataset(root      = args.img_root,
					transform = train_transform,
					train     = True,
					download  = True),
			batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(
			dataset(root      = args.img_root,
					transform = test_transform,
					train     = False,
					download  = True),
			batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

	nets = {'snet':snet, 'tnet':tnet}
	criterions = {'criterionCls':criterionCls, 'criterionFSP':criterionFSP}

	# the first stage, FSP for initilization
	print('The first stage, FSP for initilization......')
	for epoch in range(1, 51):
		epoch_start_time = time.time()

		adjust_lr_init(optimizer, epoch)

		# train one epoch
		train_init(train_loader, nets, optimizer, criterions, epoch)
		epoch_time = time.time() - epoch_start_time
		print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

		# save model
		print('saving models......')
		save_name = 'fsp_init_r{}_r{}_{:>03}.ckp'.format(args.t_name[6:], args.s_name[6:], epoch)
		save_name = os.path.join(args.save_root, 'checkpoint', save_name)
		if epoch == 1:
			save_checkpoint({
				'epoch': epoch,
				'snet': snet.state_dict(),
				'tnet': tnet.state_dict(),
			}, save_name)
		else:
			save_checkpoint({
				'epoch': epoch,
				'snet': snet.state_dict(),
			}, save_name)

	# the second stage, softmax training
	print('The second stage, softmax training......')
	for epoch in range(1, args.epochs+1):
		epoch_start_time = time.time()

		adjust_lr(optimizer, epoch)

		# train one epoch
		train(train_loader, nets, optimizer, criterions, epoch)
		epoch_time = time.time() - epoch_start_time
		print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

		# evaluate on testing set
		print('testing the models......')
		test_start_time = time.time()
		test(test_loader, nets, criterions)
		test_time = time.time() - test_start_time
		print('testing time is {:02}h{:02}m{:02}s'.format(*transform_time(test_time)))

		# save model
		print('saving models......')
		save_name = 'fsp_r{}_r{}_{:>03}.ckp'.format(args.t_name[6:], args.s_name[6:], epoch)
		save_name = os.path.join(args.save_root, 'checkpoint', save_name)
		save_checkpoint({
			'epoch': epoch,
			'snet': snet.state_dict(),
		}, save_name)

def train_init(train_loader, nets, optimizer, criterions, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	fsp_losses = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']

	criterionFSP = criterions['criterionFSP']

	snet.train()

	end = time.time()
	for idx, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		pre_s, rb1_s, rb2_s, rb3_s, _ = snet(img)
		pre_t, rb1_t, rb2_t, rb3_t, _ = tnet(img)

		fsp1_s = fsp_matrix(pre_s, rb1_s)
		fsp2_s = fsp_matrix(rb1_s, rb2_s)
		fsp3_s = fsp_matrix(rb2_s, rb3_s)
		fsp1_t = fsp_matrix(pre_t, rb1_t)
		fsp2_t = fsp_matrix(rb1_t, rb2_t)
		fsp3_t = fsp_matrix(rb2_t, rb3_t)
		fsp_loss  = (criterionFSP(fsp1_s, fsp1_t.detach()) +
					 criterionFSP(fsp2_s, fsp2_t.detach()) +
					 criterionFSP(fsp3_s, fsp3_t.detach())) / 3.0 * args.lambda_fsp
		loss = fsp_loss

		fsp_losses.update(fsp_loss.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if idx % args.print_freq == 0:
			print('Epoch[{0}]:[{1:03}/{2:03}] '
				  'Time:{batch_time.val:.4f} '
				  'Data:{data_time.val:.4f}  '
				  'FSP:{fsp_losses.val:.4f}({fsp_losses.avg:.4f})'.format(
				  epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
				  fsp_losses=fsp_losses))

def train(train_loader, nets, optimizer, criterions, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	cls_losses = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']

	criterionCls = criterions['criterionCls']

	snet.train()

	end = time.time()
	for idx, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		_, _, _, _, output_s = snet(img)

		cls_loss = criterionCls(output_s, target)

		loss = cls_loss

		prec1, prec5 = accuracy(output_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if idx % args.print_freq == 0:
			print('Epoch[{0}]:[{1:03}/{2:03}] '
				  'Time:{batch_time.val:.4f} '
				  'Data:{data_time.val:.4f}  '
				  'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
				  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
				  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
				  epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
				  cls_losses=cls_losses, top1=top1, top5=top5))

def test(test_loader, nets, criterions):
	cls_losses = AverageMeter()
	fsp_losses = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	snet = nets['snet']
	tnet = nets['tnet']

	criterionFSP = criterions['criterionFSP']
	criterionCls = criterions['criterionCls']

	snet.eval()

	end = time.time()
	for idx, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		with torch.no_grad():
			pre_s, rb1_s, rb2_s, rb3_s, output_s = snet(img)
			pre_t, rb1_t, rb2_t, rb3_t, _        = tnet(img)

		cls_loss = criterionCls(output_s, target)

		fsp1_s = fsp_matrix(pre_s, rb1_s)
		fsp2_s = fsp_matrix(rb1_s, rb2_s)
		fsp3_s = fsp_matrix(rb2_s, rb3_s)
		fsp1_t = fsp_matrix(pre_t, rb1_t)
		fsp2_t = fsp_matrix(rb1_t, rb2_t)
		fsp3_t = fsp_matrix(rb2_t, rb3_t)
		fsp_loss  = (criterionFSP(fsp1_s, fsp1_t.detach()) +
					 criterionFSP(fsp2_s, fsp2_t.detach()) +
					 criterionFSP(fsp3_s, fsp3_t.detach())) / 3.0 * args.lambda_fsp

		prec1, prec5 = accuracy(output_s, target, topk=(1,5))
		cls_losses.update(cls_loss.item(), img.size(0))
		fsp_losses.update(fsp_loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [cls_losses.avg, fsp_losses.avg, top1.avg, top5.avg]
	print('Cls: {:.4f}, FSP: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

def adjust_lr_init(optimizer, epoch):
	scale   = 0.1
	lr_list = [args.lr*scale] * 30
	lr_list += [args.lr*scale*scale] * 10
	lr_list += [args.lr*scale*scale*scale] * 10

	lr = lr_list[epoch-1]
	print('epoch: {}  lr: {}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list = [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	print('epoch: {}  lr: {}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def fsp_matrix(fm1, fm2):
	if fm1.size(2) > fm2.size(2):
		fm1 = F.avg_pool2d(fm1, 2)

	assert fm1.size(2) == fm2.size(2)
	assert fm1.size(3) == fm2.size(3)

	fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
	fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

	fsp = torch.bmm(fm1, fm2) / fm1.size(2)

	return fsp

if __name__ == '__main__':
	main()