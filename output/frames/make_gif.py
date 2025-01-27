import imageio
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--gif_name', type=str, default='output.gif')
args = parser.parse_args()


def make_gif(image_list, gif_name):
    images = []
    files = os.listdir(image_list)
    files = [file for file in files if file.endswith('.png')]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for file in files:
        file_path = os.path.join(image_list, file)
        images.append(imageio.imread(file_path))
    imageio.mimsave(gif_name, images, fps=20)
    return


if __name__ == "__main__":
    make_gif(args.dir, args.gif_name)
