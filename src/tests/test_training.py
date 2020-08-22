import io
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

from genotype.genotype import RandomArchitectureGenerator


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
                          test_loss,
                          epoch * len(testloader) + i)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    return


model_save_dir = Path(r"G:\OneDrive - UNSW\University\Postgraduate\Year 3 Trimester 2\COMP9417\Major Project\src\saved_models")
torch.cuda.empty_cache()



batch_size = 8
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_pth = r"G:\OneDrive - UNSW\University\Postgraduate\Year 3 Trimester 2\COMP9417\Major Project\src\mnist_train"
test_pth = r"G:\OneDrive - UNSW\University\Postgraduate\Year 3 Trimester 2\COMP9417\Major Project\src\mnist_test"

trainset = datasets.MNIST(train_pth, train=True, download=False, transform=transform)
testset = datasets.MNIST(test_pth, train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

images, labels = next(iter(trainloader))
in_size = (int(images.shape[1]), int(images.shape[2]), int(images.shape[2]))
rag = RandomArchitectureGenerator(
    prediction_classes=10,
    min_depth=2,
    max_depth=3,
    image_size=in_size[1],
    input_channels=in_size[0],
    min_nodes=2
)

rag.get_architecture()
cont = rag.controller()
rag.show()
def plot_imgs(images, batch_size, predit):
    fig = plt.figure()
    for i in range(len(images)):
        plt.subplot(batch_size/4, 4, i + 1)
        plt.tight_layout()
        plt.imshow(images[i][0].cpu(), cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            predit.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    return fig

def plot_to_image(figure):
    #From https://www.tensorflow.org/tensorboard/image_summaries
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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
images= images.to(device).cuda(device)
labels= labels.to(device).cuda(device)
grid = torchvision.utils.make_grid(images).to(device).cuda()

cont.update_device(device)
cont.to(device)
running_loss = 0.0

lr = 0.01
optimizer = optim.Adam(cont.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(cont.parameters(), lr=lr, momentum=0.9)

batch_tt = np.empty(shape=(10,))
k = 0
num_epochs = 2
start_t = default_timer()
break_flag = False
writer = SummaryWriter()
for epoch in range(num_epochs - 1):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):
        batch_st = default_timer()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = [d.to(device).cuda(device) for d in data]
        inputs.requires_grad = True

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = cont(inputs)
        loss = criterion(outputs, labels)#.to(device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        batch_et = default_timer()

        batch_tt[k] = batch_et - batch_st

        k += 1

        if i % 10 == 9:  # every 10 batches...

            # ...log the running loss
            writer.add_scalar('Training loss',
                              running_loss / 10,
                              epoch * len(trainloader) + i)
            writer.add_scalar('Learning rate', +
            optimizer.param_groups[0]['lr'],
                              epoch * len(trainloader) + i)
            writer.add_scalar('Average batch time',
                              np.mean(batch_tt).item(),
                              epoch * len(trainloader) + i)

            running_loss = 0.0
            batch_tt = np.empty(shape=(10,))
            k = 0

            test(cont, testloader, writer, epoch, i)

time_str = datetime.now().strftime('%Y%m%d_%H%M')
folder_name = f'{time_str}_TEN_l_{str(np.round(loss.item(),3)).replace(".","_")}'
model_name = 'evo_net.pt'
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

fig = plt.figure()
for i in range(len(images)):
    plt.subplot(batch_size/4, 4, i + 1)
    plt.tight_layout()
    plt.imshow(images[i][0].cpu(), cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])

fig.savefig(out_dir/'predict.png')

plt.show()

