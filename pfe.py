import torch
import numpy as np

def get_gradient(A, y, x):
    (n ,d) = A.size()
    tmp = y - torch.mm(A, x.view(d, -1))
    return - torch.mm(A.t(), tmp.view(n, -1)) / np.float(n)

def get_loss(A, y, x):
    (n ,d) = A.size()
    return torch.sum(torch.pow(y - torch.mm(A, x.view(d, -1)), 2))

def nnls(A, y, num_epoch, batch_size, learning_rate, adagrad = False, 
             use_GPU = False, D_vec = None, D_vec_weight = 0.1, verbose = False):
    (n, d) = A.shape
    scale = np.sqrt(6) / np.sqrt(np.float(d))
    x = (np.random.rand(d) * 2 - 1.0) * scale
    A_torch = torch.FloatTensor(A)
    y_torch = torch.FloatTensor(y)
    x_torch = torch.FloatTensor(x)
    if D_vec is not None:
        D_vec_torch = torch.FloatTensor(D_vec)
    if use_GPU:
        A_torch = A_torch.cuda()
        y_torch = y_torch.cuda()
        x_torch = x_torch.cuda()
        if D_vec is not None:
            D_vec_torch = D_vec_torch.cuda()
    for epoch in range(num_epoch):
        if adagrad:
            sum_g_square = 0
        if verbose: print "Epoch:", epoch
        loss = get_loss(A_torch, y_torch, x_torch)
        if verbose: print "Current loss is:", loss
        seq = np.random.permutation(n)
        train_sample_list = np.array_split(seq, len(seq) / batch_size)
        for sample_ind in train_sample_list:
            if use_GPU:
                sample_ind_torch = torch.cuda.LongTensor(sample_ind)
            else:
                sample_ind_torch = torch.LongTensor(sample_ind)
            A_sub = torch.index_select(A_torch, 0, sample_ind_torch)
            y_sub = y_torch[sample_ind_torch]
            g = get_gradient(A_sub, y_sub, x_torch)
            if D_vec is not None:
                g += D_vec_weight * D_vec_torch.view(d, -1)
            if adagrad:
                sum_g_square = sum_g_square + torch.pow(g, 2)
                x_torch = x_torch - learning_rate * g / torch.sqrt(sum_g_square)
#                 print sum_g_square
            else:
                x_torch = x_torch - learning_rate / np.float(epoch + 1) * g
            x_torch = torch.clamp(x_torch, min = 0.0)
    #         loss = get_loss(A, y, x)
    #         print "Current loss is:", loss
    if use_GPU:
        x_torch = x_torch.cpu()
    return (x_torch.numpy(), loss)