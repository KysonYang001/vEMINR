from skimage import io
import argparse
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yz', help='path to yz reconstructed volume',)
    parser.add_argument('--xz', help='path to yz reconstructed volume')
    parser.add_argument('--save', help='your path to save reconstructed image', default='')
    args = parser.parse_args()

    yz = io.imread(args.yz)
    xz = io.imread(args.xz)

    mean = (yz*1.0+xz*1.0) / 2
    mean = mean.astype(np.uint8)

    io.imsave(args.save, mean)



