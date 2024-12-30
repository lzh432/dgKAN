import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=100000, type=int, help='number of interactions in a testing batch')
    parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=180, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=128, type=int, help='embedding size')
    parser.add_argument('--transgnn_layer', default=2, type=int, help='number of TransGNN_gnn layers')
    parser.add_argument('--num_head', default=4, type=int, help='Multihead number of transformer layer')
    parser.add_argument('--dropout', default=0, type=float, help='Ratio of transformer layer dropout')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--edgeSampRate', default=0.75, type=float, help='ratio of edges to keep')
    parser.add_argument('--topk', default=25, type=int, help='k of topk')
    parser.add_argument('--alpha', default=0.2, type=int, help='ratio of attention sample to keep')
    parser.add_argument('--mult', default=1e-1, type=float, help='multiplication factor')
    parser.add_argument('--data', default='DrugBank', type=str, help='name of dataset')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='0', type=int, help='indicates which gpu to use')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed')
    parser.add_argument('--iteration', type=int, default='1', help='iteration')
    parser.add_argument('--is_debug', type=bool, default=True, help='is_debug')
    parser.add_argument('--dense', action='store_true', default=False, help='dense')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='if set, use validation mode which splits all relations into \
	                          train/val/test and evaluate on val only;\
	                        otherwise, use testing mode which splits all relations into train/test')
    return parser.parse_args()


args = ParseArgs()
