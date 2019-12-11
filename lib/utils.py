import os
import pdb
import torch

def save_session(model, optim, save_dir, note, epoch):
    # note0 loss note1 lr
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, note, str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)
    # save the model and optimizer state
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(optim.state_dict(), os.path.join(path, 'optim.pth'))
    print('Successfully saved model into {}'.format(path))

def load_session(model, optim, args):
    try:
        start_epoch = int(args.load_dir.split('/')[-1]) + 1
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.pth')))
        optim.load_state_dict(torch.load(os.path.join(args.load_dir, 'optim.pth')))
        for param_group in optim.param_groups:
            param_group['lr'] = args.lr
        print('Successfully loaded model from {}'.format(args.load_dir))
    except Exception as e:
        pdb.set_trace()
        print('Could not restore session properly, check the load_dir')

    return model, optim, start_epoch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count

