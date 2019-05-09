'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from OCC_CIFAR10 import OCC_CIFAR10
from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = OCC_CIFAR10(root='data', train=True, download=False, transform=transform_train)
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = OCC_CIFAR10(root='data', train=False, download=False, transform=transform_test)
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = VGG('VGG19')
net = ResNet18()
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
	checkpoint = torch.load('./checkpoint/ckpt_occluded.t7')
	net.load_state_dict(checkpoint['net'])
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets, occ) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		total_train_loss = train_loss/(batch_idx+1)
		total_train_Acc = 100.*(correct/total)
	print('train Loss:','{:3f}'.format(total_train_loss),'Train Acc:', '{:3f}'.format(total_train_Acc))
	# file = open('log/train.txt','a')

	# 	# file.write('train Loss:'),'{:3f}'.format(total_train_loss),'Train Acc:', '{:3f}'.format(total_train_Acc)) 
	# 	# file.write('\n')
	# file.write('epoch:')
	# file.write('{:d}'.format(epoch))
	# file.write('   ')
	# file.write('train Loss:')
	# file.write('{:3f}'.format(total_train_loss))
	# file.write('   ')
	# file.write('Train Acc:')
	# file.write('{:3f}'.format(total_train_Acc)) 
	# file.write('\n')

		# progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
		#    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	correct_2 = 0
	
	with torch.no_grad():
		t1 = time.time()
		t = 0
		for batch_idx, (inputs, targets, occ) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			t2 = time.time()
			outputs = net(inputs)
			t3 = time.time()
			t_1 = t3-t2
			t += t_1
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
			total_test_loss = test_loss/(batch_idx+1)
			total_test_Acc = 100.*(correct/total)
			#_, predicted_2 = torch.topk(outputs,2)

			#correct_2 += predicted_2[:,0].eq(targets).sum().item()
			#correct_2 += predicted_2[:,1].eq(targets).sum().item()
			#total_test_Acc2 = 100.*(correct_2/total)
		t4 = time.time()
		t_2 = t4 - t1
		print('test Loss:', '{:3f}'.format(total_test_loss)
		,'   ','test Acc:', '{:3f}'.format(total_test_Acc),'   ',
		 'time_batch:', '{:3f}'.format(t_1),
		 '   ', 'time_epoch:', '{:3f}'.format(t_2), '   ', 'time_epoch/net:', '{:3f}'.format(t))


		 #'test_top2 Acc:', '{:3f}'.format(total_test_Acc2))
		# file = open('log/test.txt','a')
		# 	#file.write('test Loss:', '{:3f}'.format(total_test_loss),'   ','test Acc:', '{:3f}'.format(total_test_Acc))
		
		# file.write('epoch:')
		# file.write('{:d}'.format(epoch))
		# file.write('   ')	
		# file.write('Test Loss:')
		# file.write('{:3f}'.format(total_test_loss))
		# file.write('   ')
		# file.write('test Acc:')
		# file.write('{:3f}'.format(total_test_Acc)) 
		# file.write('\n')

			# progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			#    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:
		print('Saving..Saving..Saving..Saving..Saving..')
		state = {
			'net': net.state_dict(),
			'acc': acc,
			'epoch': epoch,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, './checkpoint/ckpt_ef.t7')
		best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
	train(epoch)
	test(epoch)
