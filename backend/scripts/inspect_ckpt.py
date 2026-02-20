import torch

def inspect_checkpoint(checkpoint_path):
    print(f"🔍 Inspecting: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Check Worker Dimensions (fused_ih shape [hidden*2, input])
        # In v10, fused_ih is in backbone of each worker
        w_ih = ckpt.get("swarm.workers.M5.backbone.fused_ih.weight")
        if w_ih is not None:
             hidden_dim = w_ih.shape[0] // 2
             input_dim = w_ih.shape[1]
             print(f"  🔹 Worker: hidden_dim={hidden_dim}, input_dim={input_dim}")
        
        # Check Coordinator Dimensions
        c_ih = ckpt.get("coordinator.cfc.fused_ih.weight")
        if c_ih is not None:
            c_hidden = c_ih.shape[0] // 2
            print(f"  🔹 Coordinator: hidden_dim={c_hidden}")
            
        print("\nWeights sample keys:")
        for k in list(ckpt.keys())[:10]:
            print(f"  - {k}: {ckpt[k].shape}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_checkpoint("logs/models/best_model_v9.pth")
