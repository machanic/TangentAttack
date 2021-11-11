import pickle
import torch
import glog as log
import os.path as osp

class GradDataset(torch.utils.data.Dataset):
    def __init__(self, tuple_fnames, use_true_grad, pre_load=True):
        self.tuple_fnames = tuple_fnames
        assert len(self.tuple_fnames) > 0

        # save parameters
        self.use_true_grad = use_true_grad
        self.pre_load = pre_load
        self.grad_key = 'true_grad' if use_true_grad else 'grad'

        # get image ids
        self.image_ids = [int(tuple_fname.split('/')[-2].split('.')[0].split('-')[-1])
                          for tuple_fname in self.tuple_fnames]

        # get image and grad shape, we will reshape data accordingly
        item = torch.load(self.tuple_fnames[0], map_location='cpu')
        self.image_shape = item['adv_image'].shape
        assert len(self.image_shape) == 4 and self.image_shape[0] == 1
        self.image_shape = self.image_shape[1:]
        self.grad_shape = item[self.grad_key].shape
        assert len(self.grad_shape) == 4 and self.grad_shape[0] == 1
        self.grad_shape = self.grad_shape[1:]

        if self.pre_load:
            # pre-load all tuple data into memory, in order to reduce io load after start training
            self.samples = dict()
            self.samples['adv_image'] = list()
            self.samples['grad'] = list()
            self.samples['image_and_label'] = dict()
            for index in range(len(self.tuple_fnames)):
                if self.image_ids[index] in self.samples['image_and_label']:
                    # image and label of this tuple have been loaded before
                    adv_image, _, _, grad = self.load_from_disk(index, no_pkl=True)
                else:
                    # image and label of this tuple have not been visited
                    adv_image, image, label, grad = self.load_from_disk(index, no_pkl=False)
                    self.samples['image_and_label'][self.image_ids[index]] = (image.clone(), label.clone())
                self.samples['adv_image'].append(adv_image)
                self.samples['grad'].append(grad)
                n = index + 1
                if n % 1000 == 0 or n == len(self.tuple_fnames):
                    log.info('Pre-load tuples: {} / {} done'.format(n, len(self.tuple_fnames)))
        else:
            pass

    def load_from_disk(self, index, no_pkl=False):
        tuple_fname = self.tuple_fnames[index]
        image_id = self.image_ids[index]

        # load pkl for clean image and label
        if not no_pkl:
            pkl_fname = osp.join('/'.join(tuple_fname.split('/')[:-3]), 'image-id-{}.pkl'.format(image_id))
            assert osp.exists(pkl_fname)
            with open(pkl_fname, 'rb') as f:
                data = pickle.load(f)
            image = torch.FloatTensor(data['unperturbed'])
            label = torch.LongTensor([data['original_class']])[0]  # scalar tensor
            assert image.ndimension() == 4 and image.shape[0] == 1
            assert label.numel() == 1 and label.ndimension() == 0
            image = image.view(self.image_shape)
        else:
            image = label = None

        # load tuple
        item = torch.load(self.tuple_fnames[index], map_location='cpu')
        adv_image, grad = item['adv_image'].view(self.image_shape), item[self.grad_key].view(self.grad_shape)

        # return results
        return adv_image, image, label, grad

    def __getitem__(self, index):
        if self.pre_load:
            # get result from memory directly
            image_id = self.image_ids[index]
            adv_image = self.samples['adv_image'][index]
            image, label = self.samples['image_and_label'][image_id]
            grad = self.samples['grad'][index]
            return adv_image, image, label, grad
        else:
            # read from the disk
            return self.load_from_disk(index, no_pkl=False)

    def __len__(self):
        return len(self.tuple_fnames)
