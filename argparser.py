import argparse
import datetime


ATTRS_DEFAULT = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
    'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache',
    'No_Beard', 'Pale_Skin', 'Young']


def train_parser(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_path', dest='root_path',
                        default='/home/incentive/Music/Final Checking')
    parser.add_argument('--image_container_path', dest='image_container_path',
                        default='data/CelebA')
    parser.add_argument('--train_manifest_file_path',
                        dest='train_manifest_file_path',
                        default='data/train_manifest.txt')
    parser.add_argument('--validation_manifest_file_path',
                        dest='validation_manifest_file_path',
                        default='data/valid_manifest.txt')
   

    parser.add_argument('--img_size', dest='img_size', type=int, default=384)
    parser.add_argument('--attrs', dest='attrs', default=ATTRS_DEFAULT,
                        nargs='+', help='attributes to learn')
    parser.add_argument('--load_epoch', dest='load_epoch', type=str,
                        default='latest')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=200, help='# of epochs')

    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=64)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16,
                        help='# of sample images')

    parser.add_argument('--save_interval', dest='save_interval', type=int,
                        default=500)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int,
                        default=500)
    dt = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default=dt)
    return parser.parse_args(args)


def test_parser(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_path', dest='root_path',
                        default='/home/incentive/Music/Final Checking')
    parser.add_argument('--image_container_path', dest='image_container_path',
                        default='data/CelebA')
    parser.add_argument('--test_manifest_file_path',
                        dest='test_manifest_file_path',
                        default='data/test_manifest.txt')
    parser.add_argument('--weights_path', type=str,
                        help='weights path name here...',
                        default='src/weights/weights.149.pth')
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=384)
    parser.add_argument('--thres_int', dest='thres_int', type=float,
                        default=0.5)
    parser.add_argument('--attrs', dest='attrs', default=ATTRS_DEFAULT,
                        nargs='+', help='attributes to learn')
    
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int,default=5)
   
    return parser.parse_args(args)


def inference_parser(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='root path type here',
                        default='/home/incentive/Music/Final Checking')
    parser.add_argument('--image_folder_name', type=str,
                        help='image folder name here...',
                        default='custom')
    parser.add_argument('--image_fname', type=str,
                        help='image file name here...',
                        default='tom_cruise.jpg')
    parser.add_argument('--weights_path', type=str,
                        help='weights path name here...',
                        default='src/weights/weights.149.pth')
    
    
    # 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
    #'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache',
    #'No_Beard', 'Pale_Skin', 'Young']

    parser.add_argument('--input_img_attr', nargs='+',
                        help='input image attributes list',
                        default=[0,0,1,0,1,1,0,1,0,0,1,0,1])
    parser.add_argument('--index', nargs='+',
                        help='index of change attributes list',
                        default=[1,7])
    
    parser.add_argument('--thres_int', dest='thres_int', type=float,
                        default=0.5)
    parser.add_argument('--img_size', dest='img_size', type=int, default=384)
    
    return parser.parse_args(args)


# def preproces_parser(args=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--root_path', type=str, help='root path type here',
#                         default='/home/incentive/Desktop/FINAL Project/ojt-project/dataset')
#     parser.add_argument('--manifest-file-name', type=str,
#                         help='attribute test file name',
#                         default='list_attr_celeba.txt')
#     return parser.parse_args(args)

