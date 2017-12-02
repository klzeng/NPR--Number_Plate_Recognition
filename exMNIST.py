import numpy as np
import cv2
import struct


def extract_labels(mnist_label_file_path, label_file_path):
    with open(mnist_label_file_path, "rb") as mnist_label_file:
        # 32 bit integer magic number
        mnist_label_file.read(4)
        # 32 bit integer number of items
        mnist_label_file.read(4)
        # actual test label
        label_file = open(label_file_path, "w")
        label = mnist_label_file.read(1)
        while label:
            label_file.writelines(str(label[0]) + "\n")
            label = mnist_label_file.read(1)
        label_file.close()


def extract_images(images_file_path, images_save_folder):
    # images_file_path = "./t10k-images-idx3-ubyte"
    with open(images_file_path, "rb") as images_file:
        # 32 bit integer magic number
        images_file.read(4)
        # 32 bit integer number of images
        images_file.read(4)
        # 32 bit number of rows
        images_file.read(4)
        # 32 bit number of columns
        images_file.read(4)
        # every image contain 28 x 28 = 784 byte, so read 784 bytes each time
        count = 1
        image = np.zeros((28, 28, 1), np.uint8)
        image_bytes = images_file.read(784)
        while image_bytes:
            image_unsigned_char = struct.unpack("=784B", image_bytes)
            for i in range(784):
                image.itemset(i, image_unsigned_char[i])
            image_save_path = "./%s/%d.png" % (images_save_folder, count)
            cv2.imwrite(image_save_path, image)
            print(count)
            image_bytes = images_file.read(784)
            count += 1

if __name__ == '__main__':
    extract_images('-images-idx3-ubyte', 'MNIST')