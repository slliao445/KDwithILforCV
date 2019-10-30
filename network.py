import torch
import torch.nn as nn
import util
import math
import torch.nn.functional as F
from torch.nn import init
import DenseNet as densenet


def define_tsnet(name, num_class, cuda=True):
	if name == 'resnet20':
		net = resnet20(num_class=num_class)
	elif name == 'resnet32':
		net = resnet32(num_class=num_class)
	elif name == 'resnet110':
		net = resnet110(num_class=num_class)
	elif name == 'resnet1202':
		net = resnet1202(num_class=num_class)
	elif name == 'resnext29_16_64':
		net = resnext29_16_64(num_class=num_class)
	elif name == 'resnext29_8_64':
		net = resnext29_8_64(num_class=num_class)
	elif name == 'densenetBC100':
		net = densenet.DenseNet100(num_classes=num_class)
	elif name == 'densenetBC250':
		net = densenet.DenseNet250(num_classes=num_class)
	else:
		raise Exception('model name does not exist.')

	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	util.print_network(net)
	return net


class resblock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(resblock, self).__init__()
		self.downsample = (in_channels != out_channels)
		if self.downsample:
			self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
			self.ds    = nn.Sequential(*[
							nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
							nn.BatchNorm2d(out_channels)
							])
		else:
			self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
			self.ds    = None
		self.bn1   = nn.BatchNorm2d(out_channels)
		self.relu  = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2   = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample:
			residual = self.ds(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, num_class, num_blocks):
		super(ResNet, self).__init__()
		self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1     = nn.BatchNorm2d(16)
		self.relu    = nn.ReLU(inplace=True)

		self.res1 = self.make_layer(resblock, num_blocks[0], 16, 16)
		self.res2 = self.make_layer(resblock, num_blocks[1], 16, 32)
		self.res3 = self.make_layer(resblock, num_blocks[2], 32, 64)

		self.avgpool = nn.AvgPool2d(8)
		self.fc      = nn.Linear(64, num_class)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def make_layer(self, block, num, in_channels, out_channels):
		layers = [block(in_channels, out_channels)]
		for i in range(num-1):
			layers.append(block(out_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		pre = self.conv1(x)
		pre = self.bn1(pre)
		pre = self.relu(pre)

		rb1 = self.res1(pre)
		rb2 = self.res2(rb1)
		rb3 = self.res3(rb2)

		out = self.avgpool(rb3)
		out = out.view(out.size(0), -1)
		out = self.fc(out)

		return pre, rb1, rb2, rb3, out


def resnet20(num_class):
	return ResNet(num_class, [3, 3, 3])


def resnet32(num_class):
	return ResNet(num_class, [5, 5, 5])


def resnet110(num_class):
	return ResNet(num_class, [18, 18, 18])


def resnet1202(num_class):
	return ResNet(num_class, [200, 200, 200])


'''
ResNext optimized for the Cifar dataset, as specified in
https://arxiv.org/pdf/1611.05431.pdf
'''


class ResNeXtBlock(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels, cardinality, base_width, stride=1, downsample=None):
		super(ResNeXtBlock, self).__init__()

		D = int(math.floor(out_channels * (base_width / 64.0)))
		C = cardinality

		self.conv1 = nn.Conv2d(in_channels, D * C, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(D * C)

		self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
		self.bn2 = nn.BatchNorm2d(D * C)

		self.conv3 = nn.Conv2d(D * C, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channels * 4)

		self.downsample = downsample

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = F.relu(self.bn1(out), inplace=True)

		out = self.conv2(out)
		out = F.relu(self.bn2(out), inplace=True)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample:
			residual = self.downsample(x)

		out += residual
		out = F.relu(out, inplace=True)

		return out


class ResNeXt(nn.Module):
	def __init__(self, block, depth, cardinality, base_width, num_classes):
		super(ResNeXt, self).__init__()

		# Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
		assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
		layer_blocks = (depth - 2) // 9

		self.cardinality = cardinality
		self.base_width = base_width
		self.num_classes = num_classes

		self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
		self.bn_1 = nn.BatchNorm2d(64)

		self.inplanes = 64
		self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
		self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
		self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
		self.avgpool = nn.AvgPool2d(8)
		self.classifier = nn.Linear(256 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				init.kaiming_normal(m.weight)
				m.bias.data.zero_()

		def _make_layer(self, block, planes, blocks, stride=1):
			downsample = None
			if stride != 1 or self.inplanes != planes * block.expansion:
				downsample = nn.Sequential(
					nn.Conv2d(self.inplanes, planes * block.expansion,
							  kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(planes * block.expansion),
				)

			layers = []
			layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample))
			self.inplanes = planes * block.expansion
			for i in range(1, blocks):
				layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))

			return nn.Sequential(*layers)

		def forward(self, x):
			pre = self.conv_1_3x3(x)
			pre = F.relu(self.bn_1(pre), inplace=True)

			rb1 = self.stage_1(pre)
			rb2 = self.stage_2(rb1)
			rb3 = self.stage_3(rb2)

			out = self.avgpool(rb3)
			out = out.view(out.size(0), -1)
			out = self.classifier(out)

			return pre, rb1, rb2, rb3, out


def resnext29_16_64(num_classes=10):
	"""Constructs a ResNeXt-29, 16*64d model for CIFAR-10 (by default)
	Args:
	num_classes (uint): number of classes
	"""
	model = ResNeXt(ResNeXtBlock, 29, 16, 64, num_classes)
	return model


def resnext29_8_64(num_classes=10):
	"""Constructs a ResNeXt-29, 8*64d model for CIFAR-10 (by default)
	Args:
	num_classes (uint): number of classes
	"""
	model = ResNeXt(ResNeXtBlock, 29, 8, 64, num_classes)
	return model


'''
DenseNet
'''


# for train_ft (factor transfer)
def define_paraphraser(k, cuda=True):
	net = paraphraser(k)
	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	util.print_network(net)
	return net

class paraphraser(nn.Module):
	def __init__(self, k):
		super(paraphraser, self).__init__()
		self.encoder = nn.Sequential(*[
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Conv2d(64, int(64*k), kernel_size=3, stride=1, padding=1, bias=False)
			])
		self.decoder = nn.Sequential(*[
				nn.BatchNorm2d(int(64*k)),
				nn.ReLU(),
				nn.Conv2d(int(64*k), 64, kernel_size=3, stride=1, padding=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
			])

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		out = self.decoder(z)
		return z, out

def define_translator(k, cuda=True):
	net = translator(k)
	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	util.print_network(net)
	return net

class translator(nn.Module):
	def __init__(self, k):
		super(translator, self).__init__()
		self.encoder = nn.Sequential(*[
				nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Conv2d(64, int(64*k), kernel_size=3, stride=1, padding=1, bias=False)
			])

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		return z
