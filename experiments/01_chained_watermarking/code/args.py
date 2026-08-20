
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    # ============================ Model arguments ===================================
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'news'],
                        help="dataset name")
    parser.add_argument('--num_classes', type=int, default=10, choices=[4, 10, 100, 200, 1000],
                        help="the classes of dataset")
    parser.add_argument('--train_batch_size', type=int, default=256, choices=[128, 256, 512, 1024],
                        help="the train batch size")
    parser.add_argument('--test_batch_size', type=int, default=128, choices=[128, 128, 256, 512, 1024],
                        help="the test batch size")
    parser.add_argument('--image_size', type=int, default=32, choices=[28, 32, 64, 224], help="the size of image")
    parser.add_argument('--image_channel', type=int, default=3, help="the channel of image")
    parser.add_argument('--embed_dim', type=int, default=128, help="the dimension of text data embedding")
    parser.add_argument('--max_len', type=int, default=200, help="the maximum length of text data sentence")
    parser.add_argument('--networks', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'vgg16', 'alexnet',
                                                                             'wideresnet', 'vit', 'textcnn', 'bert'], help="model name")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate for the model")
    parser.add_argument('--epochs', type=int, default=10, help="the epochs of training")
    parser.add_argument('--max_epochs', type=int, default=12, help="the max epochs of training")
    parser.add_argument('--n_model', type=int, default=1, help="the times of the training")
    parser.add_argument('--log_interval', type=int, default=100, choices=[10, 20, 30, 50, 100], help="the log_interval")

    # ============================ Watermark arguments ===================================
    parser.add_argument('--is_embed_watermark', type=int, default=1, choices=[1, 0],
                        help="if need to embed the watermark?")
    parser.add_argument('--watermark_bit', type=int, default=1024, choices=[128, 256, 512, 1024, 2048],
                        help="the length of the watermark")
    parser.add_argument('--redund_rate', type=int, default=1, choices=[1, 2, 4, 8, 12],
                        help="redundancy rate")
    parser.add_argument('--key_len', type=int, default=2048, choices=[512, 1024, 2048, 4096],
                        help="the length of the watermark Key")
    parser.add_argument('--alpha', type=float, default=0.1, help="adjustable parameter alpha")
    parser.add_argument('--beta', type=float, default=0.1, choices=[0.1, 0.3, 0.5],
                        help="adjustable parameter beta")
    parser.add_argument('--gamma', type=float, default=0.1, choices=[0.1, 0.3, 0.5],
                        help="adjustable parameter gamma")
    parser.add_argument('--weights_batch_size', type=int, default=10, choices=[10, 20, 50], help="batch size of weights")
    parser.add_argument('--wm_methods', type=str, default='RIGA', choices=['EDNN', 'RIGA', 'FedIPR'],
                        help="model name")

    ############################################### add dp noise###############################
    parser.add_argument('--is_add_dp', type=int, default=1, help="choice if add dp")
    parser.add_argument('--epsilon', type=int, default=5, choices=[0.5, 1, 2, 5, 666], help="the density of dp")

    ###############################################OFA/WMA attack config####################
    parser.add_argument('--is_fine_tue', type=int, default=0, choices=[0, 1], help="if fine tune attack?")
    parser.add_argument('--fine_watermark_bit', type=int, default=512, choices=[128, 256, 512, 1024, 2048],
                        help="the length of the fine_tuned watermark")
    parser.add_argument('--fine_learning_rate', type=float, default=0.001, help="learning rate of the fine tune")
    parser.add_argument('--fine_tue_epochs', type=int, default=20, help="the epochs of fine tune")

    ############################################### pruning attack config#############################
    parser.add_argument('--is_pruning', type=int, default=0, choices=[0, 1], help="if pruning attack?")
    parser.add_argument('--prun_watermark_bit', type=int, default=512, choices=[128, 256, 512, 1024, 2048],
                        help="the length of the pruned watermark")
    parser.add_argument('--pruning_rate', type=float, default=0.2, choices=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                                                            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98],
                        help="pruning_rate of pruning attack")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser_args()
    print(args)