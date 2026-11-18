import numpy as np
import argparse

import torch
from models import cls_model, build_model
from utils import create_dir, rotate_points, batched_predict, viz_cloud

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--arch', type=str, default='pointnet', help='pointnet or pointnet2')
    parser.add_argument('--rot_deg', type=float, default=0.0, help='Rotate test inputs by this many degrees (Q3)')
    parser.add_argument('--rot_axis', type=str, default='x', help='Rotation axis: x, y or z')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--viz', action='store_true', help='Save a gif of object --i with its predicted class')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = build_model('cls', args.arch).to(args.device)
    
    # Load Model Checkpoint
    ckpt_dir = './checkpoints/cls' + ('' if args.arch == 'pointnet' else '_' + args.arch)
    model_path = '{}/{}.pt'.format(ckpt_dir, args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    test_data = rotate_points(test_data, args.rot_deg, args.rot_axis)

    # ------ TO DO: Make Prediction ------
    pred_label = batched_predict(model, test_data, args.batch_size, args.device)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    if args.viz:
        names = ['chair', 'vase', 'lamp']
        gt, pr = int(test_label[args.i]), int(pred_label[args.i])
        print("object {}: gt={} pred={}".format(args.i, names[gt], names[pr]))
        viz_cloud(test_data[args.i], "{}/cls_{}_{}_gt-{}_pred-{}.gif".format(
            args.output_dir, args.exp_name, args.i, names[gt], names[pr]), args.device)
