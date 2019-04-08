# change from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/cifar.py
from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
	import cPickle as pickle
else:
	import pickle

from vision import VisionDataset
#from .utils import download_url, check_integrity


class OCC_CIFAR10(VisionDataset):
	"""`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
	Args:
		root (string): Root directory of dataset where directory
			``cifar-10-batches-py`` exists or will be saved to if download is set to True.
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set.
		transform (callable, optional): A function/transform that takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.
	"""
	base_folder = 'occluded-cifar-10-batches-py'
	#url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
	#filename = "cifar-10-python.tar.gz"
	#tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
	train_list = [
		'data_batch_1',
		'data_batch_2',
		'data_batch_3',
		'data_batch_4',
		'data_batch_5',
	]

	test_list = ['test_batch']
	# meta = {
	#     'filename': 'batches.meta',
	#     'key': 'label_names',
	#     'md5': '5ff9c542aee3614f3951f8cda6e48888',
	# }

	def __init__(self, root, train=True,
				 transform=None, target_transform=None,
				 download=False):

		super(OCC_CIFAR10, self).__init__(root)
		self.transform = transform
		self.target_transform = target_transform

		self.train = train  # training set or test set

		if download:
			self.download()

		if self.train:
			downloaded_list = self.train_list
		else:
			downloaded_list = self.test_list

		self.data = []
		self.targets = []
		self.occludes = []

		# now load the picked numpy arrays
		for file_name in downloaded_list:
			file_path = os.path.join(self.root, self.base_folder, file_name)
			print(file_path)
			with open(file_path, 'rb') as f:
				entry = pickle.load(f)
				self.data.append(entry['data'])
				self.targets.extend(entry['labels'])
				self.occludes.append(entry['occludes'])
			print('extend:',len(self.targets))
		self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
		#self.targets = np.vstack(self.targets).reshape(1,1,50000)
		#self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
		#self.data = self.data.transpose((0, 1, 2,3))
		# self.targets_1 = []
		# for i in range(5):
		# 	for j in range(10000):
		# 		self.targets_1[0].append(self.targets[i][j])
		# self.targets = self.targets_1
		#	self.targets[0].append(self.targets[i+1]) 
	# def _load_meta(self):
	# 	path = os.path.join(self.root, self.base_folder, self.meta['filename'])
	# 	if not check_integrity(path, self.meta['md5']):
	# 		raise RuntimeError('Dataset metadata file not found or corrupted.' +
	# 						   ' You can use download=True to download it')
	# 	with open(path, 'rb') as infile:
	# 		if sys.version_info[0] == 2:
	# 			data = pickle.load(infile)
	# 		else:
	# 			data = pickle.load(infile, encoding='latin1')
	# 		self.classes = data[self.meta['key']]
	# 	self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target, occludes) where target is index of the target class.
		"""
		# In here. len(data):10000, len(data[index]=3; len(target)=1, len(targets[0])=10000 )
		img, target= self.data[index], self.targets[index]
		#print('data:',self.data[index],'data_len:', len(self.data),'data[0]:',self.data[0][index])
		#print('target:', self.targets,'target_len:', len(self.targets),'target[0]:',self.targets[0][index])
		#print('data:',len(self.data), 'target:', len(self.targets))
		#img, target, occludes = self.data[index], self.targets[0][index], self.occludes[0][index]

		#print(index)
		# # doing this so that it is consistent with all other datasets
		# # to return a PIL Image

		#img = Image.fromarray(img)

		# if self.transform is not None:
		# 	img = self.transform(img)

		# if self.target_transform is not None:
		# 	target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)


	def download(self):
		import tarfile


		download_url(self.url, self.root, self.filename, self.tgz_md5)

		# extract file
		with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
			tar.extractall(path=self.root)

	def extra_repr(self):
		return "Split: {}".format("Train" if self.train is True else "Test")


# class CIFAR100(CIFAR10):
# 	"""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
# 	This is a subclass of the `CIFAR10` Dataset.
# 	"""
# 	base_folder = 'cifar-100-python'
# 	url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
# 	filename = "cifar-100-python.tar.gz"
# 	tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
# 	train_list = [
# 		['train', '16019d7e3df5f24257cddd939b257f8d'],
# 	]

# 	test_list = [
# 		['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
# 	]
# 	meta = {
# 		'filename': 'meta',
# 		'key': 'fine_label_names',
# 		'md5': '7973b15100ade9c7d40fb424638fde48',
# 	}
