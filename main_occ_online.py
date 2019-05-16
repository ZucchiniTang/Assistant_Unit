'''Train Occluded CIFAR10 with PyTorch.modle:OCC_VGG16_v4_0 '''


from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import shutil

from models import *
import torchvision
import torchvision.transforms as transforms
#from OCC_CIFAR10 import OCC_CIFAR10
from OCC_CIFAR10_1 import CIFAR10
import argparse
import transforms
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--epochs', default=300, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
						help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
					help='path to save checkpoint (default: checkpoint)')				
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if not os.path.isdir(args.checkpoint):
	mkdir_p(args.checkpoint)

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
	 #transforms.RandomCrop(32, padding=4),
	 transforms.RandomHorizontalFlip(),
	 transforms.ToTensor(),
	 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	 transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.1, ),
])

transform_test = transforms.Compose([
	 transforms.ToTensor(),
	 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	 transforms.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.1, ),
])
# trainset without transform is a tuple(didn't shuffle yet) [data, target, occludes]
trainset = CIFAR10(root='data', train=True, download=False, transform=transform_train)
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = CIFAR10(root='data', train=False, download=False, transform=transform_test)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
net = VGG_OCC('VGG16_v4_0')
# net = ResNet18_occ()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True

if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.t7')
	net.load_state_dict(checkpoint['net'])
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']

criterion1 = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	correct_occ = 0
	for batch_idx, (inputs, targets, occludes) in enumerate(trainloader):
		occludes = occludes.type(dtype=torch.long)

		inputs, targets,occludes = inputs[0].to(device), targets.to(device), occludes.to(device)
		
		outputs = net(inputs)
		outputs_occ = outputs[:,0:2]
		outputs_class = outputs[:,2:12]
		loss1 = criterion1(outputs_class, targets)
		loss2 = criterion1(outputs_occ, occludes)
		loss = 4.0*loss1+loss2
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs_class.max(1)
		_, predicted_occ = outputs_occ.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		correct_occ += predicted_occ.eq(occludes).sum().item()



		total_train_loss = train_loss/(batch_idx+1)
		total_train_Acc = 100.*(correct/total)
		total_occ_train_Acc = 100.*(correct_occ/total)
	print('train Loss:','{:3f}'.format(total_train_loss),'Train Acc:', '{:3f}'.format(total_train_Acc), 'train_occ_Acc:', '{:3f}'.format(total_occ_train_Acc)  )
	return total_train_loss, total_train_Acc

def test(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	correct_occ = 0
	with torch.no_grad():
		t1 = time.time()
		t = 0
		
		for batch_idx, (inputs, targets, occludes)in enumerate(testloader):
			#inputs, targets = inputs.to(device), targets.to(device)
			occludes = occludes.type(dtype=torch.long)
			inputs, targets,occludes = inputs[0].to(device), targets.to(device), occludes.to(device)
			t2 = time.time()
			outputs = net(inputs)
			t3 = time.time()
			t_1 = t3-t2
			t += t_1
			outputs_occ = outputs[:,0:2]
			outputs_class = outputs[:,2:12]
			
			loss1 = criterion1(outputs_class, targets)
			loss2 = criterion1(outputs_occ, occludes)
			loss = 4.0*loss1+loss2

			test_loss += loss.item()
			_, predicted = outputs_class.max(1)
			_, predicted_occ = outputs_occ.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			correct_occ += predicted_occ.eq(occludes).sum().item()
			
			total_test_loss = test_loss/(batch_idx+1)
			total_test_Acc = 100.*(correct/total)
			total_occ_test_Acc = 100.*(correct_occ/total)

		t4 = time.time()
		t_2 = t4 - t1
		print('test Loss:', '{:3f}'.format(total_test_loss),'   ',
			'test Acc:', '{:3f}'.format(total_test_Acc),'   ',  'time_batch:', '{:3f}'.format(t_1),'   ', 'time_epoch:', '{:3f}'.format(t_2), '   ', 'time_epoch/net:', '{:3f}'.format(t))
	return total_test_loss, total_test_Acc

def adjust_learning_rate(optimizer, epoch):
	global state
	if epoch in args.schedule:
		state['lr'] *= args.gamma
		for param_group in optimizer.param_groups:
			param_group['lr'] = state['lr']

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
	filepath = os.path.join(checkpoint, filename)
	torch.save(state, filepath)
	if is_best:
		shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

for epoch in range(start_epoch, args.epochs):
	adjust_learning_rate(optimizer, epoch)
	
	print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
	
	train_loss, train_acc = train(epoch)
	test_loss, test_acc = test(epoch)
	
	# append logger file
	logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
	
	# save checkpoint    
	if test_acc > best_acc:
		is_best = True
		print('Saving..Saving..Saving..Saving..Saving..')
		save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': net.state_dict(),
				'acc': test_acc,
				'best_acc': best_acc,
				'optimizer' : optimizer.state_dict(),
			}, is_best, checkpoint=args.checkpoint)
		best_acc = test_acc
		
logger.close()
logger.plot()
savefig(os.path.join(args.checkpoint, 'log.eps'))

print('Best acc:')
print(best_acc)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
