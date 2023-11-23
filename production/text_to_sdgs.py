import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='System options')

    # add arguments -----------------------------------------

    parser.add_argument(
        'arguemnt name',
        metavar='source type',
        choices=['list of choices for help'],
        type='argument data-type',
        help='help string',
    )

    # boolean arguemtns
    parser.add_argument(
        '--cpu',
        help='Set true to force CPU',
        default=False,
        action='store_true'
    )

    args = vars(parser.parse_args())