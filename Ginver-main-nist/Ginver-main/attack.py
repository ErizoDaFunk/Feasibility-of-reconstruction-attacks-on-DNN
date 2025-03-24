
from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from Model import AdversarialNet, Net
import numpy as np
import random
import torchvision.utils as vutils
from torchvision import datasets
import os, shutil
from utils import TV

# 以test_images作为训练集

# 定义Train函数
def train(classifier, inversion, device, data_loader, optimizer, epoch):
    classifier.eval()
    inversion.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, relu=2)
        reconstruction = inversion(prediction)
        # blackbox
        # with torch.no_grad():
        #     grad = torch.zeros_like(reconstruction)
        #     num = 0
        #     for j in range(1, 50):
        #         random_direction = torch.randn_like(reconstruction)
        #
        #         new_pic1 = reconstruction + step * random_direction
        #         new_pic2 = reconstruction - step * random_direction
        #
        #         target1 = classifier(new_pic1, relu=2)
        #         target2 = classifier(new_pic2, relu=2)
        #
        #         loss1 = F.mse_loss(target1, prediction)
        #         loss2 = F.mse_loss(target2, prediction)
        #
        #         num = num + 2
        #         grad = loss1 * random_direction + grad
        #         grad = loss2 * -random_direction + grad
        #
        #     grad = grad / (num * step)
        #     # grad = grad.squeeze(dim=0)
        # #loss_TV = 3*TV(reconstruction)
        # #loss_TV.backward(retain_graph=True)
        # reconstruction.backward(grad)
        # optimizer.step()
        reconstruction_prediction = classifier(reconstruction, relu=2)
                                                
        # Asegúrate de que las dimensiones coincidan
        if reconstruction_prediction.size() != prediction.size():
            reconstruction_prediction = F.interpolate(reconstruction_prediction, size=prediction.size()[2:])                                     
                                                
        loss_TV = TV(reconstruction)
        loss_mse = F.mse_loss(reconstruction_prediction, prediction)
        loss = loss_mse + 0.05 * loss_TV
        loss.backward()
        optimizer.step()

    print(' Train epoch {} '.format(epoch))


    # test
def test(classifier, inversion, device, data_loader):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, relu=2)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

    mse_loss /= len(data_loader.dataset) * 64 * 64
    # print('\nTest inversion model on test set: Average MSE loss: {:.4f}\n'.format(mse_loss))
    return mse_loss

# record
def record(classifier, inversion, device, data_loader, epoch, msg, num, loss, mode, layer_name, end_epoch):
    classifier.eval()
    inversion.eval()

    plot = True
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, relu=2)
            reconstruction = inversion(prediction)

            truth = data[0:num]
            inverse = reconstruction[0:num]
            out = torch.cat((inverse, truth))

            vutils.save_image(out, '../ImagResult/blackbox/'+mode+'/'+ layer_name +'/{}_{}_{:.4f}.png'.format(msg.replace(" ", ""), epoch, loss), normalize=False)
            if epoch != end_epoch-1:
                vutils.save_image(reconstruction[0], '../ImagResult/blackbox/'+mode+'/'+ layer_name + '/final_inverse.png', normalize=False)
                vutils.save_image(data[0], '../ImagResult/blackbox/'+mode+'/'+layer_name + '/origin.png',normalize=False)
            if epoch == end_epoch-1:
                vutils.save_image(reconstruction[0], '../ImagResult/blackbox/'+mode+'/'+ layer_name + '/final_epoch.png',
                                normalize=False)
            break



def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Attack')
    parser.add_argument('--layer', type=int, default=2, metavar='N',
                        help='layer to attack (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--no-mps', action='store_true', default=False,
    #                     help='disables macOS GPU training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    print(args)


    layer = args.layer
    layer_name = "relu" + str(layer) + "_all_white_64/"
    flag = "relu" + str(layer) + "_all_white_64"
    seed = args.seed
    mode = "train"
    step = args.gamma 
    end_epoch = args.epochs
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    learning_rate = args.lr

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:

        device = torch.device("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    else:
        device = torch.device("cpu")



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs('../ImagResult/blackbox/'+mode+'/'+layer_name, exist_ok=True)
    os.makedirs('../ModelResult/blackbox/'+mode+'/'+layer_name, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Loading data...")

    train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data', train=False, transform=transform)

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size} 

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    train_loader = torch.utils.data.DataLoader(train_set,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    print("Data loaded.")

    classifier = Net().to(device)

    print("Classifier loaded.")

    inversion = AdversarialNet(nc=1, ngf=128, nz=128).to(device)

    print("Inversion loaded.")

    optimizer = optim.Adam(inversion.parameters(), lr=learning_rate, betas=(0.5, 0.999), amsgrad=True)

    # # Load classifier
    # path = "../ModelResult/classifier/classifier_32.pth"
    # classifier.load_state_dict(torch.load(path, map_location=device))

    # # Load inversion
    # path = '../ModelResult/blackbox/'+layer+'/inversion.pth'
    # best_mse_loss = 0.0600
    # begin_epoch = 1

    # try:
    #     checkpoint = torch.load(path, map_location=device)
    #     inversion.load_state_dict(checkpoint['model'])
    #     begin_epoch = checkpoint['epoch']
    #     best_mse_loss = checkpoint['best_mse_loss']
    #     print("=> loaded inversion checkpoint '{}' (epoch {}, best_mse_loss {:.4f})".format(path, begin_epoch, best_mse_loss))
    # except:
    #     print("=> load inversion checkpoint '{}' failed".format(path))

    # target_mse_loss = best_mse_loss - 0.0005




    # Load classifier
    path = "../ModelResult/classifier/classifier_32.pth"


    checkpoint = torch.load(path, map_location=device)
    classifier.load_state_dict(checkpoint)

    # Load inversion
    path = '../ModelResult/blackbox/'+layer_name+'/inversion.pth'
    best_mse_loss = 0.0600
    begin_epoch = 1

    print("Loading inversion model...")

    try:
        checkpoint = torch.load(path, map_location=device)
        inversion.load_state_dict(checkpoint['model'])
        begin_epoch = checkpoint['epoch']
        best_mse_loss = checkpoint['best_mse_loss']
        print("=> loaded inversion checkpoint '{}' (epoch {}, best_mse_loss {:.4f})".format(path, epoch, best_mse_loss))
    except:
        print("=> load inversion checkpoint '{}' failed".format(path))

    target_mse_loss = best_mse_loss - 0.0005

    print("Inversion model loaded.")

    for epoch in range(begin_epoch, end_epoch):
        print("Epoch: ", epoch)
        train(classifier, inversion, device, train_loader, optimizer, epoch)
        mse_loss = test(classifier, inversion, device, train_loader)

        print('Epoch: {} Average MSE loss: {:.4f}'.format(epoch, mse_loss))

        if mse_loss < target_mse_loss:
            target_mse_loss = mse_loss - 0.0005
            best_mse_loss = mse_loss
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mse_loss': best_mse_loss
            }
            torch.save(state, '../ModelResult/blackbox/'+mode+'/'+layer_name+'/inversion.pth')
            print('\nTest inversion model on test set: Average MSE loss: {}_{:.4f}\n'.format(epoch, mse_loss))
            record(classifier, inversion, device, test_loader, epoch, flag+"_same", 32, mse_loss, mode, layer_name, end_epoch)
            # record(classifier, inversion, device, test_loader2, epoch, flag+"_differ", 32, mse_loss)
        if epoch == end_epoch-1 :
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mse_loss': best_mse_loss
            }
            if(args.save_model):
                torch.save(state, '../ModelResult/blackbox/'+mode+'/'+layer_name+'/final_inversion.pth')
            record(classifier, inversion, device, test_loader, epoch, flag + "_same", 32, mse_loss, mode, layer_name, end_epoch)
            # record(classifier, inversion, device, test_loader2, epoch, flag+"_differ", 32, mse_loss)

if __name__ == '__main__':
    main()