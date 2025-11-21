import numpy as np
import argparse

from torch.utils.data import DataLoader

import torch
from models import cls_model, trans_cls_model
from utils import create_dir, viz_cloud

from pytorch3d.transforms import Rotate

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--indices', type=str, default=None, help="specify index of the objects to visualize, seperate values with ,")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')

    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--rotate', type=float, default=None, help='Rotates input about x axis by value if given')

    parser.add_argument('--transform', action='store_true', help='Use flag if evaluating transform model')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_success', type=int, default=6, help='Number of successful predictions to visualize')
    parser.add_argument('--num_failure', type=int, default=0, help='Number of failed predictions to visualize')
    
    parser.add_argument('--fixed_indices', action='store_true', help='Use fixed success indices: 562,397,308,434,490,413')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    if args.transform:
        chosen_model = trans_cls_model()
        checkpoint_path = './checkpoints/trans_cls/{}.pt'.format(args.load_checkpoint)
    else:
        chosen_model = cls_model()
        checkpoint_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)

    model = chosen_model
    model_path = checkpoint_path

    
    # Load Model Checkpoint
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)
    print ("successfully loaded checkpoint from {}".format(model_path))

    # FIXED: Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    test_labels = torch.from_numpy(np.load(args.test_label)).to(args.device)
    torch.cuda.empty_cache()

    if args.rotate is not None:
        R0=torch.tensor([[1., 0., 0.], [0., float(np.cos(args.rotate)), float(np.sin(args.rotate))], [0., float(-np.sin(args.rotate)), float(np.cos(args.rotate))]]).unsqueeze(0)
        R0=torch.tile(R0, (test_data.shape[0], 1, 1)).to(args.device)
        trans = Rotate(R0)
        test_data = trans.transform_points(test_data)

    # ------ TO DO: Make Prediction ------
    split_points = torch.split(test_data, args.batch_size)
    split_labels = torch.split(test_labels, args.batch_size)

    test_accuracy = 0
    pred_labels = []

    for pts_batch, lbl_batch in zip(split_points, split_labels):
        logits = model(pts_batch)
        batch_pred = torch.argmax(logits, dim=1)
        pred_labels.append(batch_pred)

    # Compute Accuracy
    pred_labels = torch.cat(pred_labels)
    test_accuracy = pred_labels.eq(test_labels.data).cpu().sum().item() / (test_labels.size()[0])
    
    if args.indices == None:
        if args.fixed_indices:
            # Hardcoded success indices (same as segmentation)
            fixed_success_indices = [562, 397, 308, 434, 490, 413]
            
            s_ind = []
            
            print(f"Using fixed success indices: {fixed_success_indices}")
            print(f"Visualizing {len(fixed_success_indices)} predictions:")
            for idx, i in enumerate(fixed_success_indices):
                cloud = test_data[i].unsqueeze(0)
                test_label = test_labels[i]
                pred_label = pred_labels[i]
                src_path = "{}/cls_s_{}_{}_{}.gif".format(args.output_dir, idx, int(test_label), pred_label)
                viz_cloud(cloud, src_path=src_path)
                s_ind.append(i)
                print(f"  Success {idx}: Index {i}, True label: {int(test_label)}, Pred: {int(pred_label)}")
            
            # Still collect failures automatically if num_failure > 0
            if args.num_failure > 0:
                failure_cases = []
                for i, items in enumerate(zip(test_data, test_labels, pred_labels)):
                    cloud, test_label, pred_label = items
                    
                    if test_label != pred_label:
                        failure_cases.append((i, cloud, test_label, pred_label))
                
                # Take first N failures
                selected_failures = failure_cases[:args.num_failure]
                
                f_ind = []
                
                print(f"\nVisualizing {len(selected_failures)} failed predictions:")
                for idx, (i, cloud, test_label, pred_label) in enumerate(selected_failures):
                    cloud = cloud.unsqueeze(0)
                    src_path = "{}/cls_f_{}_{}_{}.gif".format(args.output_dir, idx, int(test_label), pred_label)
                    viz_cloud(cloud, src_path=src_path)
                    f_ind.append(i)
                    print(f"  Failure {idx}: Index {i}, True label: {int(test_label)}, Pred: {int(pred_label)}")
                
                print("\nS indices: ", s_ind)
                print("F indices: ", f_ind)
            else:
                print("\nS indices: ", s_ind)
                print("F indices: []")
        else:
            # Original automatic selection logic
            success_indices_by_class = {0: [], 1: [], 2: []}  # chair, vase, lamp
            failure_indices_by_class = {0: [], 1: [], 2: []}
            
            # First pass: collect all success and failure indices by class
            for i, items in enumerate(zip(test_data, test_labels, pred_labels)):
                cloud, test_label, pred_label = items
                class_id = int(test_label)
                
                if test_label == pred_label:
                    success_indices_by_class[class_id].append(i)
                else:
                    failure_indices_by_class[class_id].append(i)
            
            s_ind = []
            s_class = []
            success_per_class = args.num_success // 3  # Divide equally among 3 classes
            
            for class_id in [0, 1, 2]:
                # Take first N successes for this class (consistent across runs)
                selected = success_indices_by_class[class_id][:success_per_class]
                for idx in selected:
                    s_ind.append(idx)
                    s_class.append(class_id)
            
            f_ind = []
            f_class = []
            failures_collected = 0
            
            if args.num_failure > 0:
                for class_id in [0, 1, 2]:
                    if failures_collected >= args.num_failure:
                        break
                    # Take first available failure for this class
                    if len(failure_indices_by_class[class_id]) > 0:
                        idx = failure_indices_by_class[class_id][0]
                        f_ind.append(idx)
                        f_class.append(class_id)
                        failures_collected += 1
            
            # Visualize successes
            print(f"Visualizing {len(s_ind)} successful predictions:")
            for i, (idx, class_id) in enumerate(zip(s_ind, s_class)):
                cloud = test_data[idx].unsqueeze(0)
                test_label = test_labels[idx]
                pred_label = pred_labels[idx]
                src_path = "{}/cls_s_{}_{}_{}.gif".format(args.output_dir, i, int(test_label), pred_label)
                viz_cloud(cloud, src_path=src_path)
                print(f"  Success {i}: Index {idx}, True label: {int(test_label)}, Pred: {int(pred_label)}")
            
            if args.num_failure > 0:
                print(f"\nVisualizing {len(f_ind)} failed predictions:")
                for i, (idx, class_id) in enumerate(zip(f_ind, f_class)):
                    cloud = test_data[idx].unsqueeze(0)
                    test_label = test_labels[idx]
                    pred_label = pred_labels[idx]
                    src_path = "{}/cls_f_{}_{}_{}.gif".format(args.output_dir, i, int(test_label), pred_label)
                    viz_cloud(cloud, src_path=src_path)
                    print(f"  Failure {i}: Index {idx}, True label: {int(test_label)}, Pred: {int(pred_label)}")
            
            print("\nS indices: ", s_ind)
            print("F indices: ", f_ind)
    else:
        args.indices = args.indices.split(',') 
        for i in args.indices:
            i = int(i)
            cloud = test_data[i].unsqueeze(0)
            test_label = test_labels[i] 
            pred_label = pred_labels[i]
            src_path = "{}/cls_{}_{}_{}.gif".format(args.output_dir, args.exp_name, int(test_label), pred_label)
            viz_cloud(cloud, src_path=src_path)

    print("\nTest accuracy: {}".format(test_accuracy))