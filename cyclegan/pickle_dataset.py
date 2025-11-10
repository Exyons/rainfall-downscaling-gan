import os.path
import pickle
from data.base_dataset import BaseDataset
from PIL import Image
import random
import torch
import torchvision.transforms as transforms


class PickleDataset(BaseDataset):
    """
    This dataset class can load paired data from pickle files.

    It requires two pickle files to be specified, one for each domain (A and B).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument(
            "--pickle_file_A",
            type=str,
            default="path/to/pickle_A.pkl",
            help="path to the pickle file for domain A",
        )
        parser.add_argument(
            "--pickle_file_B",
            type=str,
            default="path/to/pickle_B.pkl",
            help="path to the pickle file for domain B",
        )
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.pickle_file_A
        self.dir_B = opt.pickle_file_B
        with open(self.dir_A, "rb") as f:
            data_A = pickle.load(f)
            self.A_dates = list(data_A.keys())
            self.A_images = list(data_A.values())
        with open(self.dir_B, "rb") as f:
            data_B = pickle.load(f)
            self.B_dates = list(data_B.keys())
            self.B_images = list(data_B.values())

        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
        btoA = self.opt.direction == "BtoA"
        input_nc = (
            self.opt.output_nc if btoA else self.opt.input_nc
        )  # get the number of channels of input image
        output_nc = (
            self.opt.input_nc if btoA else self.opt.output_nc
        )  # get the number of channels of output image
        # self.transform = get_transform_rainfall(self.opt, grayscale=(self.opt.input_nc == 1))
        self.transform_A = get_transform_rainfall(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform_rainfall(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths, and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size
        A_image = self.A_images[index_A]
        # A_date = self.A_dates[index_A]

        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_image = self.B_images[index_B]
        # B_date = self.B_dates[index_B]

        # A_img = Image.fromarray(A_image[2])
        A_img = Image.fromarray(A_image)
        B_img = Image.fromarray(B_image)

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        # A_date_ts = torch.tensor(A_date.timestamp())
        # B_date_ts = torch.tensor(B_date.timestamp())

        return {"A": A, "B": B, "A_paths": str(index_A), "B_paths": str(index_B)}
        # return {'A': A, 'B': B, 'A_paths': str(index_A), 'B_paths': str(index_B), 'A_dates': A_date_ts, 'B_dates': B_date_ts}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

class NumpyToTensor:
    def __call__(self, arr):
        return torch.from_numpy(arr).float()

# * No need BaseDataset already implements it
def get_transform_rainfall(opt, grayscale=False, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    # transform_list.append(transforms.Resize([opt.crop_size, opt.crop_size], transforms.InterpolationMode.NEAREST))

    if convert:
        transform_list += [transforms.ToTensor()]
        # transform_list += [NumpyToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
