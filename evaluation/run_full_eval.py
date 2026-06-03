import argparse
import torch
import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.AuxFreeMoETransformer import AuxFreeMoeParticleTransformer 

from evaluation.data_loader import get_train_loader, get_test_loader
from evaluation.interpretability import InterpretabilityEvaluator
from evaluation.embedding_eval import EmbeddingEvaluator
from evaluation.profiler import ModelProfiler

def prepare_inputs(batch_data, device):
    
    features = batch_data['pf_features'].to(device)
    
    lorentz_vectors = batch_data['pf_vectors'].to(device)
    
    mask = batch_data['pf_mask'].to(device)
        
    return features, lorentz_vectors, mask

def load_wrapped_model(checkpoint_path, model_config, device):
    print(f"Loading wrapped model from {checkpoint_path}")
    
    model = AuxFreeMoeParticleTransformer(**model_config)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='jetclass')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./eval_results')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0)
    
    parser.add_argument('--ffn_ratio', type=int, default=2)
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=2)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    results = {}

    input_dim = 17
    num_classes = 10 if args.dataset == 'jetclass' else 2
    
    model_config = dict(
        input_dim=input_dim,
        num_classes=num_classes,
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        ffn_ratio=args.ffn_ratio,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
        for_inference=True,
        moe_num_experts=args.num_experts,
        moe_top_k=args.top_k,
        moe_capacity_factor=4,
        moe_bias_update_rate=0.001,
        moe_router_jitter=0.01,
    )

    model = load_wrapped_model(args.checkpoint, model_config, device)
    train_loader = get_train_loader(args.data_dir, args.dataset, batch_size=args.batch_size)
    test_loader = get_test_loader(args.data_dir, args.dataset, batch_size=args.batch_size)

    print("\n" + "="*40)
    print("INTERPRETABILITY ANALYSIS")
    print("="*40)

    interp_eval = InterpretabilityEvaluator(
        model=model, 
        device=device, 
        layer_idx=0, 
        input_adapter=prepare_inputs
    )
    interp_results = interp_eval.evaluate_all(test_loader)
    interp_eval.plot_interpretability(interp_results, save_dir=args.save_dir)

    # print("\n" + "="*40)
    # print("COMPUTATIONAL PROFILING")
    # print("="*40)
    
    # batch = next(iter(test_loader))
    # inputs_tuple = prepare_inputs(batch[0], device)
    # first_jet_tuple = tuple(t[0:1] for t in inputs_tuple)
    
    # profiler = ModelProfiler(model, first_jet_tuple)
    
    # results['compute'] = {
    #     'params': profiler.count_params(),
    #     'flops': profiler.count_flops(),
    #     'latency': profiler.measure_latency()
    # }
    
    # print(f"Active Params: {results['compute']['params']['active_params']}")
    # print(f"MACS: {results['compute']['flops']['total_flops']}")
    # print(f"Latency: {results['compute']['latency']['latency_ms']:.2f} ms")

    # print("\n" + "="*40)
    # print("REPRESENTATION QUALITY")
    # print("="*40)
    
    # emb_eval = EmbeddingEvaluator(model, device=device, input_adapter=prepare_inputs)
    # emb_results = emb_eval.evaluate_all(train_loader, test_loader)
    # results['embedding'] = emb_results

    # with open(os.path.join(args.save_dir, 'full_report.json'), 'w') as f:
    #     def convert(o): return float(o) if isinstance(o, (np.float32, np.float64)) else str(o)
    #     json.dump(results, f, indent=4, default=convert)

    # print(f"\nFull evaluation complete. Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()