import os
import struct

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = list(lbpath.read())
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = list(imgpath.read())
    
    images = [images[i:i + 784] for i in range(0, len(images), 784)]
    return images, labels

train_images, train_labels = load_mnist('path_to_mnist', kind='train')
test_images, test_labels = load_mnist('path_to_mnist', kind='t10k')


