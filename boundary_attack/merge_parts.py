#!/usr/bin/env python3
from glob import glob
import os
import os.path as osp
import glog as log
import sys
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='merge policy attack experiments which are split into parts')
    parser.add_argument('--dir-pattern', type=str, default='',
                        help='exp directory pattern that will be merged')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def print_args():
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


def main():
    # expand pattern
    exp_dirs = glob(args.dir_pattern)
    num_exp = len(exp_dirs)
    log.info('Found {} experiment directories using pattern {}'.format(num_exp, args.dir_pattern))

    # keep only part 0 experiments
    exp_dirs = list(filter(lambda d: 'part-0-in' in d.split('/')[-1], exp_dirs))
    num_exp_part0 = len(exp_dirs)
    log.info('Found {} part 0 experiments in all {} experiments'.format(num_exp_part0, num_exp))

    # start process part 0 experiments
    log.info('Start to merge results into part 0 experiments')
    for exp_id, exp_dir in enumerate(exp_dirs):
        log.info('Process {} / {} experiment: {}'.format(exp_id, num_exp_part0, exp_dir))

        # e.g., ['2020', '06', '18_21', '20', '53', 'L0MRieNO', 'part', '0', 'in', '60']
        split = exp_dir.split('/')[-1].split('-')
        assert len(split) == 10
        assert split[-2] == 'in'
        assert split[-3] == '0'
        assert split[-4] == 'part'
        token, num_part = split[-5], int(split[-1])
        assert len(token) == 8
        assert num_part > 0
        log.info('  token: {}, num_part: {}'.format(token, num_part))

        for part_id in range(1, num_part):
            # merge part_id experiment into part 0

            # get part_id experiment dir
            part_id_exp_dir = glob(osp.join('/'.join(exp_dir.split('/')[:-1]),
                                            '*{}-part-{}-in-{}'.format(token, part_id, num_part)))
            assert len(part_id_exp_dir) == 1
            part_id_exp_dir = part_id_exp_dir[0]
            log.info('    part {} exp dir: {}'.format(part_id, part_id_exp_dir))

            # merge results/*.pkl
            pkls_fname = glob(osp.join(part_id_exp_dir, 'results', 'image-id-*.pkl'))
            num_pkl = len(pkls_fname)
            assert num_pkl > 0
            log.info('    found {} pkls'.format(num_pkl))
            for pkl_fname in pkls_fname:
                if not osp.exists(osp.join(exp_dir, 'results', pkl_fname.split('/')[-1])):
                    shell_cmd = 'ln -s ../../{}/results/{} {}/results'.format(
                        part_id_exp_dir.split('/')[-1], pkl_fname.split('/')[-1], exp_dir)
                    os.system(shell_cmd)

            # merge results/saved_grads/image-id-*
            grads_dirname = glob(osp.join(part_id_exp_dir, 'results', 'saved_grads', 'image-id-*'))
            if len(grads_dirname) > 0:
                assert len(grads_dirname) == num_pkl > 0
                log.info('    found {} dirs for saved grads'.format(len(grads_dirname)))
                for grad_dirname in grads_dirname:
                    if not osp.exists(osp.join(exp_dir, 'results', 'saved_grads', grad_dirname.split('/')[-1])):
                        shell_cmd = 'ln -s ../../../{}/results/saved_grads/{} {}/results/saved_grads'.format(
                            part_id_exp_dir.split('/')[-1], grad_dirname.split('/')[-1], exp_dir)
                        os.system(shell_cmd)
            else:
                log.info('    no grads found')

            # merge part_id done
            log.info('   merge part {} exp done'.format(part_id))
        log.info('  merge exp {} done'.format(exp_dir))


if __name__ == '__main__':
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    args = parse_args()
    print_args()
    main()
