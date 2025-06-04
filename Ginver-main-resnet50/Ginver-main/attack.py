from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import Model
import numpy as np
import random
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
import os, shutil
from utils import TV
from torchsummary import summary
import pandas as pd
import warnings

# 以test_images作为训练集

# 定义Train函数
def train(classifier, inversion, device, data_loader, optimizer, epoch, tv_weight, layer):
    classifier.eval()
    inversion.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        # print in which batch are we from total of epoch
        if batch_idx % 100 == 0:
            print('Train epoch {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader)))

        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, layer_name=layer)

        # print("aaaa ", prediction.shape)

        reconstruction = inversion(prediction)

        # print("bbbb ", reconstruction.shape)

        reconstruction_prediction = classifier(reconstruction, layer_name=layer)

        # print("1. Reconstruction prediction shape: ", reconstruction_prediction.shape)
        # print("1. Prediction shape: ", prediction.shape)
                                                
        # Ensure the size of reconstruction_prediction matches prediction
        if reconstruction_prediction.size() != prediction.size():
            reconstruction_prediction = F.interpolate(reconstruction_prediction, size=prediction.size()[2:])

        # print("2. Reconstruction prediction shape after interpolation: ", reconstruction_prediction.shape)
        # print("2. Prediction shape: ", prediction.shape)                                     
                                                
        loss_TV = TV(reconstruction)
        loss_mse = F.mse_loss(reconstruction_prediction, prediction)
        loss = loss_mse + tv_weight * loss_TV 
        loss.backward()
        optimizer.step()

    print(' Train epoch {} '.format(epoch))


    # test
def test(classifier, inversion, device, data_loader, layer):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, layer_name=layer)
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
            prediction = classifier(data, layer_name=layer_name)
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

def needs_to_save_model(save_model_flag, layer, new_mse_loss):
    """
    Check if the model should be saved based on the save_model_flag and
    whether the current MSE loss is better than previous results
    """
    if save_model_flag:
        # Try to load the grid search results file to check for better MSE
        results_file = '../grid_search_results/grid_search_final_results.csv'
        
        # Create the results directory if it doesn't exist
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        if os.path.exists(results_file):
            try:
                # Load the grid search results
                results_df = pd.read_csv(results_file)
                
                # Check if the dataframe is empty
                if results_df.empty:
                    print(f"Grid search results file is empty")
                    return True
                
                # Filter results for the current layer
                if 'layer' in results_df.columns:
                    layer_results = results_df[results_df['layer'] == layer]
                    
                    if not layer_results.empty and 'mse_loss' in layer_results.columns:
                        # Get the best MSE loss for this layer
                        best_mse_loss = layer_results['mse_loss'].min()
                        # Check if the new MSE loss is better than the best MSE loss
                        if new_mse_loss < best_mse_loss:   
                            print(f"New MSE loss {new_mse_loss:.6f} is better than previous best {best_mse_loss:.6f}")
                            return True
                        else:
                            print(f"New MSE loss {new_mse_loss:.6f} is not better than previous best {best_mse_loss:.6f}")
                            return False
                    else:
                        print(f"No previous results found for layer {layer} in grid search")
                        return True
                else:
                    print(f"'layer' column not found in grid search results")
                    return True
            except Exception as e:
                print(f"Error reading grid search results: {e}")
                return True
            
        else:
            # Create a new CSV file with headers if it doesn't exist
            df = pd.DataFrame(columns=['mode','layer', 'batch-size', 'test-batch-size', 'epochs', 'tv-weight', 'patience', 'lr', 'gamma', 'mse_loss'])
            df.to_csv(results_file, index=False)
            print(f"Created new grid search results file {results_file}")
            return True
        
    else:
        return False
    
def get_default_params():
    """Return default parameters for training the model"""
    return {
        'mode': "whitebox",
        'layer': "maxpool",
        'batch-size': 64,
        'test-batch-size': 64,
        'epochs': 14,
        'tv-weight': 0.05,
        'patience': 3,
        'lr': 0.0002,
        'gamma': 0.7,
        'seed': 1,
        'no-cuda': False,
        'save-model': True
    }


def get_model_architecture(mode, layer):
    """Return the model architecture based on the mode and layer"""

    if mode == "blackbox":
        layer_output_channels = {
            # Early layers
            'conv1': 64,
            'relu1': 64,
            'maxpool': 64,
            # Layer 1 blocks
            'layer1': 256,
            'layer1_0': 256, 'layer1_1': 256, 'layer1_2': 256,
            # Layer 2 blocks
            'layer2': 512,
            'layer2_0': 512, 'layer2_1': 512, 'layer2_2': 512, 'layer2_3': 512,
            # Layer 3 blocks
            'layer3': 1024,
            'layer3_0': 1024, 'layer3_1': 1024, 'layer3_2': 1024, 
            'layer3_3': 1024, 'layer3_4': 1024, 'layer3_5': 1024,
            # Layer 4 blocks
            'layer4': 2048,
            'layer4_0': 2048, 'layer4_1': 2048, 'layer4_2': 2048,
        }
        nz = layer_output_channels.get(layer, 1024)  # Default to 1024 if layer not found
        return Model.ResnetInversion_Generic(nc=3, ngf=64, nz=nz)

    elif mode == "whitebox":
        if layer == "conv1":
            return Model.ResNetInversion_Conv1(nc=3)
        if layer == "maxpool":
            return Model.ResNetInversion_MaxPool(nc=3)
        # elif layer == "relu":
        #     return Model.ResNetInversion_ReLU(nc=3)
        else:
            raise ValueError(f"Unknown layer: {layer}")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ResNet EMBL Attack')

    # Get default parameters
    default_params = get_default_params()

    # Use default values from the function
    parser.add_argument('--mode', type=str, default=str(default_params['mode']), metavar='MODE',
                help=f'blackbox architecture or architecture of attacked net os known (default: {default_params["mode"]}, can be a string identifier)')
    parser.add_argument('--layer', type=str, default=str(default_params['layer']), metavar='LAYER',
                    help=f'layer to attack (default: {default_params["layer"]}, can be a string identifier)')
    parser.add_argument('--batch-size', type=int, default=default_params['batch-size'], metavar='N',
                        help=f'input batch size for training (default: {default_params["batch-size"]})')
    parser.add_argument('--test-batch-size', type=int, default=default_params['test-batch-size'], metavar='N',
                        help=f'input batch size for testing (default: {default_params["test-batch-size"]})')
    parser.add_argument('--epochs', type=int, default=default_params['epochs'], metavar='N',
                        help=f'number of epochs to train (default: {default_params["epochs"]})')
    parser.add_argument('--tv-weight', type=float, default=default_params['tv-weight'], metavar='W',
                   help=f'weight for TV loss (default: {default_params["tv-weight"]})')
    parser.add_argument('--patience', type=int, default=default_params['patience'], metavar='P',
                    help=f'early stopping patience (default: {default_params["patience"]})')
    parser.add_argument('--lr', type=float, default=default_params['lr'], metavar='LR',
                        help=f'learning rate (default: {default_params["lr"]})')
    parser.add_argument('--gamma', type=float, default=default_params['gamma'], metavar='M',
                        help=f'Learning rate step gamma (default: {default_params["gamma"]})')
    parser.add_argument('--seed', type=int, default=default_params['seed'], metavar='S',
                        help=f'random seed (default: {default_params["seed"]})')
    parser.add_argument('--no-cuda', action='store_true', default=default_params['no-cuda'],
                    help='disables CUDA training (default: %(default)s)')
    parser.add_argument('--cuda', dest='no_cuda', action='store_false',
                        help='enables CUDA training (overrides --no-cuda)')
    parser.add_argument('--save-model', action='store_true', default=default_params['save-model'],
                        help='For Saving the current Model (default: %(default)s)')
    parser.add_argument('--no-save-model', dest='save_model', action='store_false',
                        help='Disable saving the current Model')

    args = parser.parse_args()

    print(args)

    mode = args.mode
    layer = args.layer
    # layer_name = str(layer) + "_all_white_64"
    # flag = str(layer) + "_all_white_64"
    seed = args.seed
    # step = args.gamma 
    end_epoch = args.epochs
    tv_weight = args.tv_weight
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    learning_rate = args.lr
    save_model = args.save_model

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)

    print("Use cuda flag: ", not args.no_cuda)
    print("Cuda is available: ", torch.cuda.is_available())
    print("Use cuda: ", use_cuda)

    if use_cuda:
        print("Using CUDA")
        device = torch.device("cuda")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(torch.cuda.get_device_properties(device))

    else:
        print("Using CPU")
        device = torch.device("cpu")



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs('../ImagResult/'+mode+'/'+layer, exist_ok=True)
    os.makedirs('../ModelResult/'+mode+'/'+layer, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # ResNet50 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Loading Embl dataset...")

    train_dataset = ImageFolder(root='../data/GS_organized/train', transform=transform)
    test_dataset = ImageFolder(root='../data/GS_organized/test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    print(f"Data loaded: {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples.")

    # model_path = "../ModelResult/classifier/classifier.pth"
    # model = torch.load(model_path)

    # classifier = model.ResNet50EMBL(model).to(device)

    model_path = "../ModelResult/classifier/classifier.pth"
    model_weights = torch.load(model_path, map_location=device)
    classifier = Model.ResNet50EMBL(model_weights).to(device)

    # print(summary(classifier, (1, 64, 64),)) <--------------- Important line to check the model architecture

    print("Classifier loaded.")

    # Load inversion
    path = '../ModelResult/'+ mode + '/' + layer +'/inversion.pth'
    inversion = get_model_architecture(mode, layer).to(device)

    print("Inversion loaded.")

    optimizer = optim.Adam(inversion.parameters(), lr=learning_rate, betas=(0.5, 0.999), amsgrad=True)


    best_mse_loss = float('inf')
    begin_epoch = 1

    # print("Trying to load inversion model from checkpoint...")

    # try:
    #     checkpoint = torch.load(path, map_location=device)
    #     inversion.load_state_dict(checkpoint['model'])
    #     begin_epoch = checkpoint['epoch'] + 1
    #     best_mse_loss = checkpoint['best_mse_loss']
    #     print("=> loaded inversion checkpoint '{}' (epoch {}, best_mse_loss {:.4f})".format(path, checkpoint['epoch'], best_mse_loss))
    # except:
    #     print("=> load inversion checkpoint '{}' failed".format(path))
    #     begin_epoch = 1  # Comenzar desde la primera época si no hay checkpoint
    #     best_mse_loss = float('inf')

    # print("Inversion model loaded.")

    
    patience = args.patience
    
    # Variables for early stopping
    best_mse_loss = float('inf')
    patience_counter = 0

    for epoch in range(begin_epoch, end_epoch + 1):
        print("Epoch: ", epoch)
        train(classifier, inversion, device, train_loader, optimizer, epoch, tv_weight, layer)
        mse_loss = test(classifier, inversion, device, test_loader, layer)

        print('Epoch: {} Average MSE loss: {:.4f}'.format(epoch, mse_loss))

        # Logic for saving the best model
        if mse_loss < best_mse_loss:
            patience_counter = 0  
            best_mse_loss = mse_loss
            
            # Save best model
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mse_loss': best_mse_loss
            }
            torch.save(state, '../ModelResult/'+mode+'/'+ layer +'/inversion.pth')
            print('\nTest inversion model on test set: Average MSE loss: {}_{:.4f}\n'.format(epoch, mse_loss))
            # record(classifier, inversion, device, test_loader, epoch, flag+"_same", 32, mse_loss, mode, layer_name, end_epoch)
        else:
            patience_counter += 1  
            print(f'Early stopping patience: {patience_counter}/{patience}')
            
            # Early stop condition
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs. Best MSE loss: {best_mse_loss:.4f}')
                break
        
        # Save final model
        if epoch == end_epoch-1 or patience_counter >= patience:
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mse_loss': best_mse_loss
            }
            if(needs_to_save_model(save_model, layer, mse_loss)):
                print("Saving final inversion model...") 
                torch.save(state, '../ModelResult/'+mode+'/'+ layer +'/final_inversion.pth')
            # record(classifier, inversion, device, test_loader, epoch, flag + "_same", 32, mse_loss, mode, layer_name, end_epoch)

if __name__ == '__main__':
    main()