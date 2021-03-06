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

parser = argparse.ArgumentParser(description='baseline')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
parser.add_argument('--save_log', type=str, default='./logs', help='logs are saved here')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# net and dataset choosen
parser.add_argument('--data_name', type=str, default='cifar10', required=False, help='name of dataset')# cifar10/cifar100
parser.add_argument('--net_name', type=str, default='densenetBC100',required=False, help='name of basenet')

def main():
	global args
	args = parser.parse_args()
	print(args)

	if not os.path.exists(os.path.join(args.save_root,'checkpoint')):
		os.makedirs(os.path.join(args.save_root,'checkpoint'))

	if args.cuda:
		cudnn.benchmark = True

	print('----------- Network Initialization --------------')
	net = define_tsnet(name=args.net_name, num_class=args.num_class, cuda=args.cuda)
	print('-----------------------------------------------')

	# save initial parameters
	print('saving initial parameters......')

	if args.net_name[:6] == 'resnet':
		save_name = 'baseline_r{}_{:>03}.ckp'.format(args.net_name[6:], 0)
	elif args.net_name[:6] == 'resnex':
		save_name = 'baseline_rx{}_{:>03}.ckp'.format(args.net_name[7:], 0)
	elif args.net_name[:6] == 'densen':
		save_name = 'baseline_dBC{}_{:>03}.ckp'.format(args.net_name[10:], 0)

	save_name = os.path.join(args.save_root, 'checkpoint', save_name)
	save_checkpoint({
		'epoch': 0,
		'net': net.state_dict(),
	}, save_name)

	# initialize optimizer
	optimizer = torch.optim.SGD(net.parameters(),
								lr = args.lr,
								momentum = args.momentum,
								weight_decay = args.weight_decay,
								nesterov = True)

	# define loss functions
	if args.cuda:
		criterion = torch.nn.CrossEntropyLoss().cuda()
	else:
		criterion = torch.nn.CrossEntropyLoss()

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

	trainF = open(os.path.join(args.save_log, 'train.csv'), 'w')
	testF = open(os.path.join(args.save_log, 'test.csv'), 'w')

	train_start_time = time.time()

	max_test_prec_1, max_test_prec_5 = 0, 0

	for epoch in range(1, args.epochs+1):
		epoch_start_time = time.time()

		adjust_lr(optimizer, epoch)

		# train one epoch
		train(train_loader, net, optimizer, criterion, epoch, trainF)
		epoch_time = time.time() - epoch_start_time
		print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

		# evaluate on testing set
		print('testing the models......')
		test_start_time = time.time()
		test_prec_1, test_prec_5 = test(test_loader, net, criterion, testF)
		test_time = time.time() - test_start_time
		print('testing time is {:02}h{:02}m{:02}s'.format(*transform_time(test_time)))

		max_test_prec_1 = max(max_test_prec_1, test_prec_1)
		max_test_prec_5 = max(max_test_prec_5, test_prec_5)

		# save model
		print('saving models......')
		save_name = 'baseline_r{}_{:>03}.ckp'.format(args.net_name[6:], epoch)
		save_name = os.path.join(args.save_root, 'checkpoint', save_name)
		save_checkpoint({
			'epoch': epoch,
			'net': net.state_dict(),
		}, save_name)

	train_finish_time = time.time()
	train_total_time = train_start_time - train_finish_time

	trainF.close()
	testF.close()

	print('the total train time is:{:02}h{:02}m{:02}s'.format(*transform_time(train_total_time)))
	print('the max test prec@1:{:.2f} , prec@5:{:.2f}'.format(max_test_prec_1, max_test_prec_5))


def train(train_loader, net, optimizer, criterion, epoch, trainF):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	losses     = AverageMeter()
	top1       = AverageMeter()
	top5       = AverageMeter()

	net.train()

	end = time.time()
	for idx, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		_, _, _, _, output = net(img)
		loss = criterion(output, target)

		prec1, prec5 = accuracy(output, target, topk=(1,5))
		losses.update(loss.item(), img.size(0))
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
				  'loss:{losses.val:.4f}({losses.avg:.4f})  '
				  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
				  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
				  epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
				  losses=losses, top1=top1, top5=top5))

		trainF.write('{},{},{}\n'.format(epoch, loss.data[0], prec1))
		trainF.flush()


def test(test_loader, net, criterion, testF):
	losses = AverageMeter()
	top1   = AverageMeter()
	top5   = AverageMeter()

	net.eval()

	end = time.time()
	for idx, (img, target) in enumerate(test_loader, start=1):
		if args.cuda:
			img = img.cuda()
			target = target.cuda()

		with torch.no_grad():
			_, _, _, _, output = net(img)
			loss = criterion(output, target)

		prec1, prec5 = accuracy(output, target, topk=(1,5))
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))
		top5.update(prec5.item(), img.size(0))

	f_l = [losses.avg, top1.avg, top5.avg]
	print('Loss: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

	testF.write('{},{},{}\n'.format(epoch, losses.avg, top1.avg))
	testF.flush()

	return top1.avg, top5.avg


def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	print('epoch: {}  lr: {}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	main()