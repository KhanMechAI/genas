import io
import argparse
from timeit import default_timer
from pathlib import Path
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from datetime import datetime
import matplotlib.pyplot as plt
from win10toast import ToastNotifier

from genotype.genotype import RandomArchitectureGenerator

notify = ToastNotifier()


def test(network, testloader, writer, epoch, i):
    network.eval()
    test_loss = 0
    correct = 0
    test_losses = []
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(network.device).cuda(network.device)
            target = target.to(network.device).cuda(network.device)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(testloader.dataset)
        writer.add_scalar('Test loss',
                          1 + test_loss,
                          epoch * len(testloader) + i)
        writer.add_scalar('Accuracy vs log(parameters)', -test_loss * 100 / np.log(stats['Parameters']), step)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))
    return


def plot_imgs(images, batch_size, predit):
    fig = plt.figure()
    for i in range(len(images)):
        plt.subplot(batch_size / 4, 4, i + 1)
        plt.tight_layout()
        plt.imshow(images[i][0].cpu(), cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            predit.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    return fig


def plot_to_image(figure):
    # From https://www.tensorflow.org/tensorboard/image_summaries
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = np.array(Image.open(buf))
    # Add the batch dimension
    # image = tf.expand_dims(image, 0)
    return image


model_save_dir = Path(
    r"G:\OneDrive - UNSW\University\Postgraduate\Year 3 Trimester 2\COMP9417\Major Project\src\short_train_models")
data_dir = Path(r"G:\OneDrive - UNSW\University\Postgraduate\Year 3 Trimester 2\COMP9417\Major Project\src\runs_cifar")
torch.cuda.empty_cache()

batch_size = 8
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_pth = Path(
    r"G:\OneDrive - UNSW\University\Postgraduate\Year 3 Trimester 2\COMP9417\Major Project\src\cifar_10_train")
test_pth = Path(
    r"G:\OneDrive - UNSW\University\Postgraduate\Year 3 Trimester 2\COMP9417\Major Project\src\cifar_10_test")

trainset = torchvision.datasets.CIFAR10(root=train_pth, train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, )

testset = torchvision.datasets.CIFAR10(root=test_pth, train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, )

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


parser = argparse.ArgumentParser(description='Batch settings.')

parser.add_argument('--cuda', metavar='C', type=int, nargs=1,
                    help='Which GPU? cuda:X?', dest='cuda_num', default=1)
parser.add_argument('--min', metavar='m', type=int, nargs=1,
                    help='Min depth', dest='min_depth', default=4)
parser.add_argument('--max', metavar='M', type=int, nargs=1,
                    help='Max depth', dest='max_depth', default=12)
parser.add_argument('--nodes', metavar='n', type=int, nargs=1,
                    help='Min number of nodes', dest='nodes', default=6)

args = parser.parse_args()

dataiter = iter(trainloader)
images, labels = dataiter.next()
in_size = (int(images.shape[1]), int(images.shape[2]), int(images.shape[2]))

cuda_num = args.cuda_num[0]
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

lr = 0.01

num_epochs = 2
num_gens = 20
# milestones = [k for k in range(0, num_epochs*len(trainloader), 50)]
rag=[]
cont=[]
for gen in range(0, num_gens):

    cont = -1
    while cont == -1 or cont is None:
        del rag, cont
        rag = RandomArchitectureGenerator(
            prediction_classes=len(classes),
            min_depth=args.min_depth[0],
            max_depth=args.max_depth[0],
            image_size=in_size[1],
            input_channels=in_size[0],
            min_nodes=args.nodes[0],
        )
        cont = rag.get_architecture()

    rag.show()

    torch.cuda.empty_cache()

    cont.update_device(device)
    cont.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cont.parameters(), lr=lr, momentum=0.9)

    batch_tt = np.empty(shape=(10,))
    k = 0
    running_loss = 0

    start_t = default_timer()
    break_flag = False
    time_str = datetime.now().strftime('%Y%m%d%H%M')
    model_suff = cont._suffix
    comment = f'_{model_suff}'
    f_name_suff = f'_cifar'

    model_log_dir = data_dir / ''.join([time_str, comment])
    model_log_dir.mkdir(parents=True, exist_ok=True)

    stats = cont.get_stats()

    writer = SummaryWriter(log_dir=model_log_dir, filename_suffix=f_name_suff, comment=comment)

    [writer.add_scalar(k, v, gen) for k, v in stats.items()]

    with torch.cuda.device(device):
        for epoch in range(num_epochs - 1):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                batch_st = default_timer()
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = [d.to(device).cuda(device) for d in data]
                inputs.requires_grad = True

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = cont(inputs)
                loss = criterion(outputs, labels).to(device).cuda(device)
                loss.backward()
                optimizer.step()
                # scheduler.step()

                running_loss += loss.item()

                batch_et = default_timer()

                batch_tt[k] = batch_et - batch_st

                k += 1

                if i % 10 == 9:  # every 10 batches...

                    # ...log the running loss
                    step = epoch * len(trainloader) + i
                    writer.add_scalar('Training loss',
                                      running_loss / 10,
                                      step)
                    writer.add_scalar('Learning rate', +
                    optimizer.param_groups[0]['lr'],
                                      step)
                    writer.add_scalar('Average batch time',
                                      np.mean(batch_tt).item(),
                                      step)
                    writer.add_scalar('Loss per parameter count', running_loss / 10 / np.log((stats['Parameters'])),
                                      step)

                    running_loss = 0.0
                    batch_tt = np.empty(shape=(10,))
                    k = 0

                    test(cont, testloader, writer, epoch, i)

                if i > 1000:
                    break
            torch.cuda.empty_cache()

    folder_name = f'{time_str}_TEN_l_{str(np.round(loss.item(), 3)).replace(".", "_")}'
    model_name = f'evo_net_{model_suff}.pt'
    out_dir = model_save_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / model_name
    torch.save(cont, out_path)

    end_t = default_timer()

    total_t = end_t - start_t

    writer.add_scalar('Total training time',
                      total_t,
                      epoch * len(trainloader) + i)

    print('Training Complete')
    images, labels = next(iter(testloader))
    images = images.to(cont.device).cuda(cont.device)
    labels = labels.to(cont.device).cuda(cont.device)
    with torch.no_grad():
        output = cont(images)

    fig = plt.gcf()
    rag.show()
    fig.savefig(out_dir / 'arch.png')

    fig, ax = plt.subplots(int(batch_size / 4), 4)
    as_im = transforms.ToPILImage()
    for i, a in enumerate(ax.flat):
        img = images[i].cpu()
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        a.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='none')
        a.title.set_text("Prediction: {}".format(
            classes[output.data.max(1, keepdim=True)[1][i].item()]))
        a.set_xticks([])
        a.set_yticks([])

    fig.savefig(out_dir / 'predict.png')

    plt.tight_layout()
    plt.show()

    del fig, ax, a, img, images, output, labels
notify.show_toast(f"Training complete", "Your training has finished. Total time: {total_t}")
