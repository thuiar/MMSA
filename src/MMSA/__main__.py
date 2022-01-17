import argparse

from .run import MMSA_run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=False,
                        help='tune parameters ?') # TODO: refine this function
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression / classification') # TODO: remove this option & classsification configs
    parser.add_argument('--modelName', type=str, default='lf_dnn',
                        help='support lf_dnn/ef_lstm/tfn/lmf/mfn/graph_mfn/mult/misa/mlf_dnn/mtfn/mlmf/self_mm')
    parser.add_argument('--datasetName', type=str, default='sims',
                        help='support mosi/mosei/sims')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/20200506',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[1],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # TODO
