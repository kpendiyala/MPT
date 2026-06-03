import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class InterpretabilityEvaluator:
    def __init__(self, model, device='cpu', layer_idx=7, input_adapter=None):
        self.model = model
        self.device = device
        self.layer_idx = layer_idx
        self.input_adapter = input_adapter
        self.model.to(device)
        self.model.eval()
        
        self.routing_data = {'gates': [], 'features': [], 'labels': [], 'mask': []}
        self.hook_handle = None

    def _hook_fn(self, module, input, output):
        # output is router_logits. Apply sigmoid to get gate probabilities.
        self.routing_data['gates'].append(torch.sigmoid(output).detach().cpu())

    def extract_routing_data(self, dataloader, max_batches=None):
        print(f"Extracting routing data from Layer {self.layer_idx}")
        
        # 1. Attach hook to the specific block's router
        target_block = self.model.blocks[self.layer_idx]
        self.hook_handle = target_block.router.register_forward_hook(self._hook_fn)
        
        # Grab the learned bias for Aux-Free MoE evaluation
        expert_bias = target_block.expert_bias.detach().cpu()

        total_batches = len(dataloader)
        if max_batches is not None:
            total_batches = min(max_batches, total_batches) 
        
        # 2. Run Inference
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Inference", total=total_batches)):
                if max_batches is not None and i >= max_batches:
                    break
                
                inputs_dict = batch[0]
                y = batch[1].to(self.device)
                
                # Use the input adapter to format features, vectors, masks exactly as the model expects
                features, vectors, mask = self.input_adapter(inputs_dict, self.device)
                
                # Forward pass triggers the hook
                _ = self.model(features, vectors, mask)
                
                # Store inputs on CPU for later analysis
                self.routing_data['features'].append(features.detach().cpu())
                self.routing_data['labels'].append(y.detach().cpu())
                self.routing_data['mask'].append(mask.detach().cpu())

        # 3. Clean up the hook
        if self.hook_handle:
            self.hook_handle.remove()
            
        return self._process_data(expert_bias)

    def _process_data(self, expert_bias):
        # Flatten tensors: (Batch, Channels, Particles) -> (P*B, C)
        all_feats = torch.cat(self.routing_data['features'], dim=0).permute(2, 0, 1).reshape(-1, 17)
        all_masks = torch.cat(self.routing_data['mask'], dim=0).permute(2, 0, 1).reshape(-1).bool()
        
        # Expand labels to match tokens: (N, num_classes) -> (P*N, num_classes)
        num_classes = self.routing_data['labels'][0].shape[-1]
        all_labels = torch.cat(self.routing_data['labels'], dim=0).unsqueeze(0).expand(128, -1, -1).reshape(-1, num_classes)
        
        # Gates are collected as (P*N, num_experts)
        all_gates = torch.cat(self.routing_data['gates'], dim=0).reshape(-1, expert_bias.size(0))

        # Filter out padded tokens
        valid_gates = all_gates[all_masks]
        valid_feats = all_feats[all_masks]
        valid_labels = all_labels[all_masks]

        # Process PID categories (Assuming indices 6-10 match WeaverPreprocessor)
        pid_labels = []
        features_np = valid_feats.numpy()
        for i in range(len(valid_feats)):
            if features_np[i, 6] > 0: pid_labels.append('Charged Hadron')
            elif features_np[i, 7] > 0: pid_labels.append('Neutral Hadron')
            elif features_np[i, 8] > 0: pid_labels.append('Photon')
            elif features_np[i, 9] > 0: pid_labels.append('Electron')
            elif features_np[i, 10] > 0: pid_labels.append('Muon')
            else: pid_labels.append('Other/SV')

        biased_gates = valid_gates + expert_bias
        assigned_experts = biased_gates.argmax(dim=-1).numpy()

        df = pd.DataFrame({
            'Expert': assigned_experts,
            'LogPt': valid_feats[:, 0].numpy(),
            'dEta': valid_feats[:, 15].numpy(),
            'Jet_Type': valid_labels.argmax(dim=-1).numpy(),
            'PID': pid_labels
        })
        
        # Map Jet Class names (Standard JetClass order)
        jet_class_names = {
            0: 'QCD', 1: 'Hbb', 2: 'Hcc', 3: 'Hgg', 4: 'H4q', 
            5: 'Hqql', 6: 'Zqq', 7: 'Wqq', 8: 'Tbqq', 9: 'Tbl'
        }
        df['Jet_Class'] = df['Jet_Type'].map(jet_class_names).fillna('Unknown')
        
        return df

    def evaluate_all(self, test_loader, max_batches=None):
        """Main orchestrator matching EmbeddingEvaluator style."""
        df = self.extract_routing_data(test_loader, max_batches=max_batches)
        return df

    def plot_interpretability(self, df, save_dir=None):
        sns.set_theme(style="whitegrid")
        
        # 1. Expert vs PID
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Expert', hue='PID', multiple='fill', discrete=True)
        plt.title(f'Expert Assignment by Particle Type (Layer {self.layer_idx})')
        plt.ylabel('Proportion')
        plt.xticks(range(df['Expert'].nunique()))
        if save_dir: plt.savefig(os.path.join(save_dir, f'expert_vs_pid_layer_{self.layer_idx}.png'))
        else: plt.show()
        plt.close()

        # 2. Expert vs Kinematics (dEta vs LogPt)
        plt.figure(figsize=(10, 8))
        df_sample = df.sample(n=min(50000, len(df))) 
        sns.scatterplot(data=df_sample, x='dEta', y='LogPt', hue='Expert', palette='tab10', s=10, alpha=0.6)
        plt.title(f'Kinematic Routing: dEta vs LogPt (Layer {self.layer_idx})')
        if save_dir: plt.savefig(os.path.join(save_dir, f'expert_vs_kinematics_layer_{self.layer_idx}.png'))
        else: plt.show()
        plt.close()

        # 3. Expert vs Jet Class (Heatmap)
        plt.figure(figsize=(12, 6))
        heatmap_data = pd.crosstab(df['Jet_Class'], df['Expert'], normalize='index') * 100
        sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt='.1f')
        plt.title(f'Expert Utilization by Jet Class (%) (Layer {self.layer_idx})')
        plt.ylabel('Jet Class')
        plt.xlabel('Expert ID')
        if save_dir: plt.savefig(os.path.join(save_dir, f'expert_vs_jetclass_layer_{self.layer_idx}.png'))
        else: plt.show()
        plt.close()