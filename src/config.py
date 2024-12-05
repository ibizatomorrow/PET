import argparse
from utils import get_rank

parser = argparse.ArgumentParser(description='PET')

parser.add_argument("--gpus", nargs='+', type=int, default=[3],
                    help="gpus")
parser.add_argument("--batch_size", type=int, default=1,
                    help="batch size")
parser.add_argument("-d", "--dataset", type=str, default='PHO',
                    help="dataset to use")
parser.add_argument("--test", action='store_true', default=False,
                    help="load stat from dir and directly test")


# configuration for stat training
parser.add_argument("--n_epoch", type=int, default=10,
                    help="number of minimum training epochs on each time step")
parser.add_argument("--lr", type=float, default=0.0005,
                    help="learning rate")
parser.add_argument("--grad_norm", type=float, default=1.0,
                    help="norm to clip gradient to")
parser.add_argument("--negative_num", type=int, default=64,
                    help="number of negative sample")       
parser.add_argument("--adversarial_temperature", type=float, default=0.5,
                    help="adversarial temperature setting")               

# configuration for evaluating
parser.add_argument("--metric", type=list, default=['Precision@5', 'NDCG@5', 'Recall@5', 'F1@5', 'MRR@5'],
                    help="evaluating metrics")

# configuration for layers
parser.add_argument("--input_dim", type=int, default=64,
                    help="dimension of layer input")
parser.add_argument("--hidden_dims", nargs='+', type=int, default=[64, 64],
                    help="dimension list of hidden layers")
                      # note that you can specify this item using like this
                      # --hidden_dims 16 16 16 16 16 16
parser.add_argument("--message_func", type=str, default='transe',
                    help="which message_func you use")


# To simplify the command, we set True as the default for the following parameters..
parser.add_argument("--short_cut", action='store_true', default=True,
                    help="whether residual connection")
parser.add_argument("--layer_norm", action='store_true', default=True,
                    help="whether layer_norm")           

# configuration for transformer        
parser.add_argument("--layer_num", type=int, default=2,
                    help="Num of TransformerEncoderLayer") 
parser.add_argument("--num_heads", type=int, default=2,
                    help="Num of heads in multiheadattention") 
parser.add_argument("--dropout", type=int, default=0.3,
                    help="Dropout rate for transformer") 


args, unparsed = parser.parse_known_args()
if get_rank() == 0:
  print(args)  
  print(unparsed)  
