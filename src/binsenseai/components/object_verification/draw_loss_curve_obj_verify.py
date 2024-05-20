import torch
import matplotlib.pyplot as plt
import numpy as np

path = 'snapshots/resnet34_siamese_best.pth.tar'
checkpoint = torch.load(path)

train_loss_list = checkpoint['train_loss_list']
val_acc_list = checkpoint['val_acc_list']

x = np.asarray(range(len(train_loss_list)))+1
x = x/10.0
plt.figure(1)
plt.plot(x[3:],train_loss_list[3:])
plt.xlabel('Epochs')
plt.ylabel('Loss(-logp)')
plt.title('Training loss curve')
plt.savefig('Training_loss_curve.jpg')
plt.show()

x = np.asarray(range(25))+1

# Assuming val_acc_list has a length of 2
val_acc_list_padded = np.pad(val_acc_list, (0, len(x) - len(val_acc_list)), 'constant')
plt.figure(2)
plt.plot(x, val_acc_list_padded)
#plt.plot(x, val_acc_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation accuracy curve')
plt.savefig('Validation_accuracy_curve.jpg')
plt.show()