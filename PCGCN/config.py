import argparse
import torch

args = argparse.ArgumentParser()
args.add_argument('--normalize', default='yes')
args.add_argument('--dataset', default='ca-GrQc')
args.add_argument('--constraint_rate', default=1)
args.add_argument('--constraint_rate_neg', default=1)
args.add_argument('--rate', default=0.05)
args.add_argument('--model', default='GCN')
# args.add_argument('--learning_rate', default=0.01)
args.add_argument('--learning_rate', default=0.01)
args.add_argument('--pre_learning_rate', default=0.01)
args.add_argument('--epochs', default=9999)
args.add_argument('--hidden_units', default=16)
args.add_argument('--dropout', default=0.4)
args.add_argument('--weight_decay', default=0.01)
args.add_argument('--weight_decay1', default=0.01)
args.add_argument('--weight_decay2', default=0.0005)
args.add_argument('--early_stopping', default=30)
args.add_argument('--residual', default=True)
args.add_argument('--seed', default=824)
args.add_argument('--type', default='Binary')
args.add_argument('--DEVICE', default="cuda" if torch.cuda.is_available() else "cpu")
args.add_argument('--theta', default=0.5)
args.add_argument('--layers', default=64)
args.add_argument('--alpha', default=0.1)
args.add_argument('--feature_type', default='fea')
args.add_argument('--similar_matrix', default='regular_simi')
args.add_argument('--CGCN', default=0)
args = args.parse_args(args=[])






