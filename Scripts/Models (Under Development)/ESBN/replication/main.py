# python3 ./train_and_eval.py --model_name $model_name --norm_type contextnorm --lr 5e-4 --task same_diff --m_holdout 0 --epochs 50 --run $r --device $device
import numpy as np
from torch.utils.data import Dataset, DataLoader

import os
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


class seq_dataset(Dataset):
    def __init__(self, dset):
        self.seq_ind = dset['seq_ind']
        self.y = dset['y']
        self.len = self.seq_ind.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        seq_ind = self.seq_ind[idx]
        y = self.y[idx]
        return seq_ind, y


def train(model, device, optimizer, epoch, all_imgs, train_loader, task, m_holdout, model_name, run, log_interval):
    # Create file for saving training progress
    train_prog_dir = './train_prog/'
    check_path(train_prog_dir)
    task_dir = train_prog_dir + task + '/'
    check_path(task_dir)
    gen_dir = task_dir + 'm' + str(m_holdout) + '/'
    check_path(gen_dir)
    model_dir = gen_dir + model_name + '/'
    check_path(model_dir)
    run_dir = model_dir + 'run' + run + '/'
    check_path(run_dir)
    train_prog_fname = run_dir + 'epoch_' + str(epoch) + '.txt'
    train_prog_f = open(train_prog_fname, 'w')
    train_prog_f.write('batch loss acc\n')
    # Set to training mode
    model.train()
    # Iterate over batches
    for batch_idx, (seq_ind, y) in enumerate(train_loader):
        # Batch start time
        start_time = time.time()
        # Use sequence indices to slice corresponding images
        x_seq = all_imgs[seq_ind, :, :]
        # Load data to device
        x_seq = x_seq.to(device)
        y = y.to(device)
        # Zero out gradients for optimizer
        optimizer.zero_grad()
        # Run model
        if 'MNM' in model_name:
            y_pred_linear, y_pred, const_loss = model(x_seq, device)
        else:
            y_pred_linear, y_pred = model(x_seq, device)
        # Loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_pred_linear, y)
        if 'MNM' in model_name:
            loss += const_loss
        # Update model
        loss.backward()
        optimizer.step()
        # Batch duration
        end_time = time.time()
        batch_dur = end_time - start_time
        # Report prgoress
        if batch_idx % log_interval == 0:
            # Accuracy
            acc = torch.eq(y_pred, y).float().mean().item() * 100.0
            # Report
            print('[Epoch: ' + str(epoch) + '] ' + \
                  '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' + \
                  '[Loss = ' + '{:.4f}'.format(loss.item()) + '] ' + \
                  '[Accuracy = ' + '{:.2f}'.format(acc) + '] ' + \
                  '[' + '{:.3f}'.format(batch_dur) + ' sec/batch]')
            # Save progress to file
            train_prog_f.write(str(batch_idx) + ' ' + \
                               '{:.4f}'.format(loss.item()) + ' ' + \
                               '{:.2f}'.format(acc) + '\n')
    train_prog_f.close()


def _test(model, device, all_imgs, test_loader, task, m_holdout, model_name, run):
    print('Evaluating on test set...')
    # Set to eval mode
    model.eval()
    # Iterate over batches
    all_acc = []
    all_loss = []
    for batch_idx, (seq_ind, y) in enumerate(test_loader):
        # Use sequence indices to slice corresponding images
        x_seq = all_imgs[seq_ind, :, :]
        # Load data to device
        x_seq = x_seq.to(device)
        y = y.to(device)
        # Run model
        if 'MNM' in model_name:
            y_pred_linear, y_pred, const_loss = model(x_seq, device)
        else:
            y_pred_linear, y_pred = model(x_seq, device)
        # Loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_pred_linear, y)
        if 'MNM' in model_name:
            loss += const_loss
        all_loss.append(loss.item())
        # Accuracy
        acc = torch.eq(y_pred, y).float().mean().item() * 100.0
        all_acc.append(acc)
        # Report progress
        print('[Batch: ' + str(batch_idx) + ' of ' + str(len(test_loader)) + ']')
    # Report overall test performance
    avg_loss = np.mean(all_loss)
    avg_acc = np.mean(all_acc)
    print('[Summary] ' + \
          '[Loss = ' + '{:.4f}'.format(avg_loss) + '] ' + \
          '[Accuracy = ' + '{:.2f}'.format(avg_acc) + ']')
    # Save performance
    test_dir = './test/'
    check_path(test_dir)
    task_dir = test_dir + task + '/'
    check_path(task_dir)
    gen_dir = task_dir + 'm' + str(m_holdout) + '/'
    check_path(gen_dir)
    model_dir = gen_dir + model_name + '/'
    check_path(model_dir)
    test_fname = model_dir + 'run' + run + '.txt'
    test_f = open(test_fname, 'w')
    test_f.write('loss acc\n')
    test_f.write('{:.4f}'.format(avg_loss) + ' ' + \
                 '{:.2f}'.format(avg_acc))
    test_f.close()


def main():
    # Settings

    model_name = 'ESBN'
    encoder_type = 'conv'
    norm_type = 'contextnorm'
    task = 'identity_rules'
    train_gen_method = 'subsample'
    test_gen_method = 'subsample'
    n_shapes = 100
    m_holdout = 90

    train_batch_size = 10
    train_set_size = 2_000
    train_proportion = .95
    lr = 5e-4
    epochs = 10
    log_interval = 10
    test_batch_size = 100
    test_set_size = 2_000
    no_cuda = False

    run = '1'

    # Set up cuda
    device = "cpu"

    # Randomly assign objects to training or test set
    all_shapes = np.arange(n_shapes)
    np.random.shuffle(all_shapes)
    train_shapes = all_shapes
    test_shapes = all_shapes
    if m_holdout > 0:
        train_shapes = all_shapes[m_holdout:]
        test_shapes = all_shapes[:m_holdout]

    # Generate training and test sets
    # task_gen = __import__('tasks.identity_rules')
    from tasks.identity_rules import create_task, y_dim, seq_len
    print('Generating task: ' + train_gen_method + '...')
    # log.info('Generating task: ' + args.task + '...')
    train_set, test_set, train_set_size, test_set_size = create_task(
        train_shapes=train_shapes,
        test_shapes=test_shapes,
        n_shapes=n_shapes,
        m_holdout=m_holdout,
        train_set_size=train_set_size,
        test_set_size=test_set_size,
        train_proportion=train_proportion,
        train_gen_method=train_gen_method,
        test_gen_method=test_gen_method
    )

    # # Convert to PyTorch DataLoaders
    train_set = seq_dataset(train_set)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_set = seq_dataset(test_set)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

    # Load images
    all_imgs = []
    for i in range(n_shapes):
        img_fname = './imgs/' + str(i) + '.png'
        img = torch.Tensor(np.array(Image.open(img_fname))) / 255.
        all_imgs.append(img)
    all_imgs = torch.stack(all_imgs, 0)

    # Create model
    from models import ESBN

    model = ESBN.Model(
        encoder_type=encoder_type,
        norm_type=norm_type,
        y_dim=y_dim,
        seq_len=seq_len,
        task_seg=None,
    ).to(device)

    # Append relevant hyperparameter values to model name
    model_name = model_name + '_' + norm_type + '_lr' + str(lr)

    # Create optimizer
    print('Setting up optimizer...')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    print('Training begins...')
    for epoch in tqdm(range(1, epochs + 1)):
        # Training loop
        train(model, device, optimizer, epoch, all_imgs, train_loader, task, m_holdout, model_name, run, log_interval)
        # Test model
        _test(model, device, all_imgs, test_loader, task, m_holdout, model_name, run)


if __name__ == '__main__':
    main()
