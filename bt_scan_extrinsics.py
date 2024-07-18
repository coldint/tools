#!/usr/bin/env python

import sys
import logging
import argparse
import bittensor as bt

subtensor = None

def watch_extrinsics():
    last_logstr = None
    processed_this_block = -1
    last_block = -1
    add_whitespace = False
    while True:
        block = subtensor.block
        if last_block != block:
            if add_whitespace:
                print('')
                add_whitespace = False
            last_block = block
            processed_this_block = -1
        num = 0
        for idx,ext in enumerate(subtensor.substrate.retrieve_pending_extrinsics()):
            num += 1
            if idx > processed_this_block:
                processed_this_block = idx
                func = ext.value['call']['call_function']
                if func in ['set_weights','serve_axon','set_commitment','burned_register','remove_stake','transfer_allow_death','batch','add_stake']:
                    continue
                addr = ext.value['address']
                logging.info(f'block {block}-{idx}: {func} {addr}')# {ext.value}')
                add_whitespace = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='finney',
            help='Network to connect to (default local, use finney for remote)')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
            help='Increase verbosity')
    parser.add_argument('--silent', '-s', default=False, action='store_true',
            help='Decrease verbosity')

    global args
    args = parser.parse_args()

    logging.logThreads = True
    logging.logProcesses = True
    logging.logMultiprocessing = True
    logformat = logging.Formatter('%(asctime)-15s - %(message)s')
    if args.silent:
        loglevel = logging.ERROR
    else:
        loglevel = logging.DEBUG if args.verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logformat)
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    logger.addHandler(handler)

    if args.silent:
        bt.logging.off()

    global subtensor
    logging.info('connecting to network...')
    subtensor = bt.subtensor(network=args.network)
    logging.info('connected')

    watch_extrinsics()

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt as e:
        pass

