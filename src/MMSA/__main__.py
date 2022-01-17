import argparse

from .run import MMSA_run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tune', type=bool, default=False,
                        help='Whether to tune hyper parameters. Default: False')
    parser.add_argument('-m', '--model', type=str, default='lf_dnn', help='Name of model',
                        choices=['lf_dnn', 'ef_lstm', 'tfn', 'lmf', 'mfn', 'graph_mfn', 'mult', 'misa', 'mlf_dnn', 'mtfn', 'mlmf', 'self_mm'])
    parser.add_argument('-d', '--dataset', type=str, default='sims',
                        choices=['sims', 'mosi', 'mosei'], help='Name of dataset')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to config file. If not specified, default config file will be used.')
    parser.add_argument('-s', '--seeds', type=list, default=[],
                        help='List of seeds. Default: [1111, 1112, 1113, 1114, 1115]')
    parser.add_argument('-n', '--num_workers', type=int, default=8,
                        help='Number of workers used to load data. Default: 4')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Verbose level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1')
    parser.add_argument('--model_save_dir', type=str, default='',
                        help='Path to save trained models. Default: "~/MMSA/saved_models"')
    parser.add_argument('--res_save_dir', type=str, default='',
                        help='Path to save csv results. Default: "~/MMSA/results"')
    parser.add_argument('--log_dir', type=str, default='',
                        help='Path to save log files. Default: "~/MMSA/logs"')
    parser.add_argument('--gpu_ids', type=list, default=[1],
                        help='Specify which gpus to use. If a empty list is supplied, will automatically assign to the most memory-free gpu. \
                              Currently only support single gpu. Default: [0]')
    parser.add_argument('-Ft', '--feature_T', type=str, default='',
                        help='Path to text feature file. Default: ""')
    parser.add_argument('-Fa', '--feature_A', type=str, default='',
                        help='Path to audio feature file. Default: ""')
    parser.add_argument('-Fv', '--feature_V', type=str, default='',
                        help='Path to video feature file. Default: ""')
    
    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()
    MMSA_run(
        model_name=cmd_args.model,
        dataset_name=cmd_args.dataset,
        config_file=cmd_args.config,
        seeds=cmd_args.seeds,
        is_tune=cmd_args.tune,
        feature_T=cmd_args.feature_T,
        feature_A=cmd_args.feature_A,
        feature_V=cmd_args.feature_V,
        model_save_dir=cmd_args.model_save_dir,
        res_save_dir=cmd_args.res_save_dir,
        log_dir=cmd_args.log_dir,
        gpu_ids=cmd_args.gpu_ids,
        num_workers=cmd_args.num_workers,
        verbose_level=cmd_args.verbose
    )
