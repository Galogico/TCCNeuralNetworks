import torch
import torchvision
import numpy

# Quantas vezes o código vai passar pelo dataset
n_epochs = 3

# tamanho do treino e do teste respectivamente
batch_size_train = 64
batch_size_test = 1000

learning_rate = 0.01
momentum = 0.5
log_interval = 10

# Seed de números aleatórios
random_seed = 1
# Não usar algorítmos não determinísticos
torch.backends.cudnn.enabled = False
# usar a seed como a seed da NN
torch.manual_seed(random_seed)

# fazendo load do dataset de números escritos á mão
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


# aqui eu defino essa DB como exemplo
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# aqui é algo opcional que eu só mostro como é o DB
print(example_data.shape) 
# OUTPUT ESPERADO: torch.Size([1000, 1, 28, 28])
# esse output indica que existem 1000 imagens sem rgb(pixels de 0 a 1 espaço contínuo) com 28 por 28 pixels


