import argparse
from pathlib import Path

from datetime import datetime

from genotype.genotype import RandomArchitectureGenerator

img_save_dir = Path(
    r"G:\OneDrive - UNSW\University\Postgraduate\Year 3 Trimester 2\COMP9417\Major Project\src\ras_figs")

parser = argparse.ArgumentParser(description='Batch settings.')

parser.add_argument('--min', metavar='m', type=int, nargs=1,
                    help='Min depth', dest='min_depth', default=4)
parser.add_argument('--max', metavar='M', type=int, nargs=1,
                    help='Max depth', dest='max_depth', default=12)
parser.add_argument('--nodes', metavar='n', type=int, nargs=1,
                    help='Min number of nodes', dest='nodes', default=6)
parser.add_argument('--s', metavar='s', type=int, nargs=1,
                    help='Input size', dest='size', default=128)
parser.add_argument('--gen', metavar='s', type=int, nargs=1,
                    help='Number of generations', dest='gen', default=20)
args = parser.parse_args()

num_gens = args.gen[0]
rag = []
cont = []
time_str = datetime.now().strftime('%Y%m%d%H%M')
for gen in range(0, num_gens):

    cont = -1
    while cont == -1 or cont is None:
        del rag, cont
        rag = RandomArchitectureGenerator(
            prediction_classes=1,
            min_depth=args.min_depth[0],
            max_depth=args.max_depth[0],
            image_size=args.size[0],
            input_channels=1,
            min_nodes=args.nodes[0],
        )
        cont = rag.get_architecture()

    folder_name = f'{time_str}_{num_gens}_gens'
    f_name = f'{gen}_{cont._suffix}.png'

    out_dir = img_save_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f_name
    rag.show(save=out_path)
