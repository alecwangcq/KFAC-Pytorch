import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='vgg16_bn')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--optimizer', type=str, default='kfac')
parser.add_argument('--machine', type=int, default=10)

args = parser.parse_args()

vgg16_bn = ''
vgg19_bn = ''
resnet = '--depth 110'
wrn = '--depth 28 --widen_factor 10 --dropRate 0.3'
densenet = '--depth 100 --growthRate 12'

apps = {
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    'resnet': resnet,
    'wrn': wrn,
    'densenet': densenet
}


def grid_search(args):
    scripts = []
    if args.optimizer in ['kfac', 'ekfac']:
        template = 'python main.py ' \
                   '--dataset %s ' \
                   '--optimizer %s ' \
                   '--network %s ' \
                   ' --epoch 100 ' \
                   '--milestone 40,80 ' \
                   '--learning_rate %f ' \
                   '--damping %f ' \
                   '--weight_decay %f %s'

        lrs = [3e-2, 1e-2, 3e-3]
        dampings = [3e-2, 1e-3, 3e-3]
        wds = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
        app = apps[args.network]
        for lr in lrs:
            for dmp in dampings:
                for wd in wds:
                    scripts.append(template % (args.dataset, args.optimizer, args.network, lr, dmp, wd, app))
    elif args.optimizer == 'sgd':
        template = 'python main.py ' \
                   '--dataset %s ' \
                   '--optimizer %s ' \
                   '--network %s ' \
                   ' --epoch 200 ' \
                   '--milestone 60,120,180 ' \
                   '--learning_rate %f ' \
                   '--weight_decay %f %s'
        app = apps[args.network]
        lrs = [3e-1, 1e-1, 3e-2]
        wds = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

        for lr in lrs:
            for wd in wds:
                scripts.append(template % (args.dataset, args.optimizer, args.network, lr, wd, app))

    return scripts


def gen_script(scripts, machine, args):
    with open('run_%s_%s_%s.sh' % (args.dataset, args.optimizer, args.network), 'w') as f:
        for s in scripts:
            f.write('srun --gres=gpu:1 -c 6 -w guppy%d --mem=16G -p gpu \"%s\" &\n' % (machine, s))


if __name__ == '__main__':
    scripts = grid_search(args)
    gen_script(scripts, args.machine, args)

