#!/usr/bin/env python

import argparse
import bittensor as bt
import logging
import numpy as np
import sys
import json

subtensor = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subnet-id', type=int,
            help="Subnet ID to inspect")
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

def get_set_weight(subnet_id, hotkey, block_id):
    b = subtensor.substrate.get_block(block_number=block_id)
    for extrinsic in b['extrinsics']:
        call = extrinsic.value.get('call', {})
        func = call.get('call_function', '')
        call_args = call.get('call_args', [])
        if func != 'set_weights':
            continue
        if extrinsic.value.get('address', None) != hotkey:
            continue
        info = {}
        for arg in call_args:
            if arg['name'] == 'netuid':
                info['netuid'] = arg['value']
            elif arg['name'] == 'weights':
                info['weights'] = arg['value']
            elif arg['name'] == 'version_key':
                info['version'] = arg['value']
        if info.get('netuid', -1) == subnet_id:
            return info
    # Not found
    return None

def print_subnet_validators(subnet_id):
    mg = subtensor.metagraph(subnet_id, lite=True)
    stakes = mg.S
    uids = np.argsort(stakes)[::-1]
    validator_uids = [uid for uid in uids if stakes[uid]>1000]
    validator_hks = set([mg.hotkeys[uid] for uid in validator_uids])
    vtrust = mg.Tv
    hk_lastrecs = {}

    for uid in validator_uids:
        stake = stakes[uid]
        hk = mg.hotkeys[uid]
        upd = int(mg.last_update[uid])
        version = '---'
        try:
            weight_info = get_set_weight(subnet_id, hk, upd)
            if weight_info is not None:
                version = weight_info['version']
            if version > (1<<30):
                version = '0x..' + ('%x'%version)[-4:]
        except Exception as e:
            pass
        dblocks = subtensor.block - upd
        w_info = f"version {str(version):8s}, #{upd}, {dblocks*12/60:-7.01f} min ago"
        print(f"UID {uid:-4d}, stake {int(stake):-8d}, vtrust {vtrust[uid]:.03f}, {hk}, {w_info}")

def main():
    parse_args()

    global subtensor
    logging.info('connecting to network...')
    subtensor = bt.subtensor(network=args.network)
    logging.info('connected')

    print_subnet_validators(args.subnet_id)

if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt as e:
        pass
