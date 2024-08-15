#!/usr/bin/env python

import os
import sys
import json
import copy
import time
import logging
import argparse

args = None
tokenizer = None

def imports():
    """
    Delayed imports for known slow imports. This allows argument parsing to happen fast.
    """
    global torch,np,GPT2TokenizerFast,AutoTokenizer,SlicedLlamaForCausalLM
    import numpy as np
    import torch
    from transformers_llama import SlicedLlamaForCausalLM
    import transformers
    from transformers import GPT2TokenizerFast,AutoTokenizer

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

def filter_params(model,layer_from=None,layer_to=None):
    """
    Filter parameters to use for a particular range of layers.
    embed tokens is considered part of the first layer.
    lm_head and model.norm are considered part of the last layer.
    """
    params = []
    param_names_kept = set()
    param_idx_to_name = {}
    for pname, p in model.named_parameters():
        if 'embed_tokens' in pname:
            if layer_from > 0: continue
        elif 'lm_head' in pname or 'model.norm' in pname:
            if layer_to < model.config.num_hidden_layers: continue
        else:
            keep = False
            for i in range(layer_from,layer_to):
                if f'layers.{i}.' in pname:
                    keep = True
                    break
            if not keep:
                continue
        param_idx_to_name[len(params)] = pname
        param_names_kept.add(pname)
        params.append(p)
    return {
            'param_idx_to_name':param_idx_to_name,
            'param_names_kept':param_names_kept,
            'params':params,
    }

def slice_model(model,start_layers):
    slices = []
    logging.info(f'slicing model with start_layers {start_layers}')

    model_data = {}
    # strip model, restore later
    for pname, p in model.named_parameters():
        model_data[pname] = p.data
        p.data = torch.zeros(0)

    start_layers.append(model.config.num_hidden_layers)
    for i,layer_from in enumerate(start_layers[:-1]):
        layer_to = start_layers[i+1]
        logging.info(f'Generating slice #{i}: {layer_from}..{layer_to}')
        params = filter_params(model,layer_from=layer_from,layer_to=layer_to)
        model_slice = copy.deepcopy(model) # copy of empty model
        for pname,p in model_slice.named_parameters():
            if pname in params['param_names_kept']:
                p.data = model_data[pname]
        model_slice.config.start_at_layer = layer_from
        model_slice.config.return_states_at_layer = layer_to
        slices.append(model_slice)
    return slices

def evaluate_losses(model,samples):
    """
    Evaluate losses of samples on model. This is deliberately kept in, aside
    from evaluate_losses_sliced(), that is also able to evaluate a singular
    model, so that outputs can be compared.
    """
    losses = []
    for i,sample in enumerate(samples):
        if i >= args.max_samples:
            break
        token_ids_sample = sample['ids']
        ids = torch.stack([torch.tensor(token_ids_sample)]).to(args.device)
        labels = ids.clone().to(args.device)
        logging.debug(f'evaluating sample {i} of length {len(token_ids_sample)}...')
        try:
            out = model(ids, labels=labels, output_hidden_states=False)
            loss = out.loss.detach().item()
            logging.debug(f'loss: {loss}')
            losses.append(loss)
        except Exception as e:
            logging.info(f'failed to evaluate, using inf. Exception: {e}')
            losses.append(np.inf)
    return losses

def evaluate_losses_sliced(model_slices,samples):
    """
    Evaluate losses of samples on sliced model.
    """
    output_states = []
    for slice_idx,model_slice in enumerate(model_slices):
        model_slice = model_slice.to(args.device)
        model_params = model_slice.num_parameters()
        state_size = sum([s.numel() for s in output_states])
        logging.info(f'evaluating slice {slice_idx}, n_params={model_params}, state_size={state_size}...')
        losses = evaluate_losses_slice(model_slice,samples=samples,output_states=output_states)
        model_slice = model_slice.to('cpu')
        torch.cuda.empty_cache()
    return losses

def evaluate_losses_slice(model_slice,samples=None,output_states=[]):
    losses = []
    is_first_slice = model_slice.config.start_at_layer == 0
    is_last_slice = model_slice.config.return_states_at_layer == model_slice.config.num_hidden_layers
    for i,sample in enumerate(samples):
      try:
        if i >= args.max_samples:
            break
        ids = None
        if is_first_slice or is_last_slice:
            token_ids_sample = sample['ids']
            ids = torch.stack([torch.tensor(token_ids_sample)]).to(args.device)
        if is_first_slice:
            # inject token ids
            logging.debug(f'evaluating sample {i} of length {len(token_ids_sample)} in first slice...')
            outputs = model_slice.model(input_ids=ids)
            output_states.append(outputs.last_hidden_state)
        else:
            # resume with hidden states
            logging.debug(f'evaluating sample {i} in later slice...')
            outputs = model_slice.model(inputs_embeds=output_states[i])
            output_states[i] = outputs.last_hidden_state
        if is_last_slice:
            # calculate losses
            logits = model_slice.lm_head(output_states[i][0])
            logits = logits.float()
            labels = ids
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, model_slice.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
            loss_value = loss.detach().item()
            logging.debug(f'loss: {loss_value}')
            losses.append(loss_value)
      except Exception as e:
        logging.warning(f'Exception evaluating sample {i}, length {len(sample["ids"])}: {e}')
        raise e

    return losses

def load_sample_file(fn):
    """
    Load samples from a file. Support .json, .jsonl or plain text.
    """
    samples = []
    with open(fn,'r') as f:
        if 'jsonl' in fn:
            # enforce one json object per line
            for line in f:
                if len(line) == 0:
                    continue
                js = json.loads(line)
                if type(js) is dict:
                    samples.append(js)
                elif type(js) is str:
                    samples.append({'text':js})
                else:
                    raise Exception("jsonl only supports lines with objects or strings")
        elif 'json' in fn:
            # load a complete json object
            samples = json.load(f)
        else:
            # best effort line based import
            for line in f:
                if len(line) == 0:
                    continue
                try:
                    js = json.loads(line)
                    if type(js) is dict:
                        samples.append(js)
                    else:
                        samples.append({'text':str(js)})
                except:
                    samples.append({'text':line})
    return samples

def load_samples():
    samples = []
    for f in args.samples:
        if not os.path.exists(f):
            logging.error(f'ignoring non-existant sample file {f}')
            continue
        if not os.path.isfile(f):
            logging.error(f'ignoring non-file sample arg {f}')
            continue
        samples.extend(load_sample_file(f))
    if len(args.samples) and not len(samples):
        raise Exception("Failed to load any sample")
    for s in samples:
        if type(s) is not dict:
            raise Exception(f"sample {s} is not a dict")
        if 'ids' in s:
            continue
        if not 'text' in s:
            raise Exception(f"no 'text' key in sample {s}")
        s['ids'] = tokenizer(s['text'])['input_ids']
    return samples

def load_model(path, attn_implementation="flash_attention_2"):
    if args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    elif args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError("Unkown datatype {args.dtype}")

    logging.info(f"Loading model {path}, attn={attn_implementation}, dtype {args.dtype}")
    model = SlicedLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation=attn_implementation,
        torch_dtype=dtype
    )
    try:
        tokenizer_obj = AutoTokenizer.from_pretrained(path)
        logging.info('loaded tokenizer from model path')
    except:
        tokenizer_name = "Xenova/gpt-4"
        logging.info('falling back to default tokenizer: {tokenizer_name}')
        tokenizer_obj = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    return model, tokenizer_obj

def arg_parser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
            help='Model directory to process')
    parser.add_argument('samples', default=[], nargs='+',
            help='Directory with .jsonl sample files (or one or more filenames)')
    parser.add_argument('--max-samples', default=None, type=int,
            help='Keep only this maximum number of samples')
    parser.add_argument('--max-sample-len', default=100000, type=int,
            help='Maximum sample length')
    parser.add_argument('--attn', default=None, choices=['sdpa','eager','flash_attention_2'],
            help='Override attention implementation when loading model (note that eager and flash_attention_2 are compatible, but the latter requires a cuda device)')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16','float16','float32'],
            help='Select model datatype, bfloat16 is default')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
            help='Increase verbosity')
    parser.add_argument('--device', default='cuda:0',
            help='Cuda device to use')
    parser.add_argument('--start-layers', default='0',
            help='List of integers specifying layer starts for each slice (e.g. 0,4,8,12)')
    parser.add_argument('--auto-slice', metavar='N', default=None, type=int,
            help='Automatically slice model in N parts.')

    args = parser.parse_args(argv)

    if args.start_layers is not None:
        args.start_layers = [int(s) for s in args.start_layers.split(',')]

    return args

def main():
    global args, tokenizer
    args = arg_parser(sys.argv[1:])

    logging.logThreads = True
    logging.logProcesses = True
    logging.logMultiprocessing = True
    logformat = logging.Formatter('%(asctime)-15s - %(message)s')
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logformat)
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    logger.addHandler(handler)

    logging.debug('loading imports...')
    imports()
    logging.debug('done loading imports')

    logging.debug(f"Evaluating model {args.model}")

    attn_implementation = args.attn
    if attn_implementation is None:
        if args.device.startswith('cuda'):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "eager"
    t0 = time.time()
    model, tokenizer = load_model(args.model, attn_implementation=attn_implementation)
    t_loading = time.time() - t0

    if args.auto_slice is not None:
        n_layers = model.config.num_hidden_layers
        n_slices = min(args.auto_slice,n_layers)
        args.start_layers = [0]
        while len(args.start_layers)<n_slices:
            last_start = args.start_layers[-1]
            remaining_slices = n_slices-len(args.start_layers)
            remaining_layers = n_layers-last_start
            next_start = last_start + remaining_layers//(remaining_slices+1)
            args.start_layers.append(next_start)

    samples = load_samples()
    logging.debug(f'loaded {len(samples)} samples')

    with torch.no_grad():
        t0 = time.time()
        model_slices = slice_model(model,args.start_layers)
        t_slicing = time.time() - t0
        t0 = time.time()
        losses = evaluate_losses_sliced(model_slices,samples)
        t_evaluating = time.time() - t0
        # show loss sum and a few individual losses; should be identical regardless of slicing
        logging.info(f'losses: sum={sum(losses)}, {losses[:20]}...')

    logging.info(f'time stats: {t_loading:.01f}s loading, {t_slicing:.01f}s slicing, {t_evaluating:.01f}s evaluating')

if __name__ == '__main__':
    try:
        if not main():
            sys.exit(-1)
        sys.exit(0)
    except KeyboardInterrupt as e:
        pass
