#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
{Description}
"""

from __future__ import print_function
import os
import sys
import argparse

from utils import create_data_lists


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--voc07_path', help="VOC2007 data path",
                        type=str)
    parser.add_argument('--voc12_path', help="VOC2012 data path",
                        type=str)
    parser.add_argument('-o', '--out_dir', help="Output directory",
                        type=str)

    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    create_data_lists(args.voc07_path, voc12_path=None, \
                      output_folder=args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

