import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from model.data import Tokenizer
from model.memmap_data import create_memmap_dataloaders
from model.model import PhyloLM

def run_profile(args):
    device = torch.device("cuda")
    tokenizer = Tokenizer()
    
    print("Initializing Dataloaders...")
    train_iter, _ = create_memmap_dataloaders(
        memmap_dir=args.memmap_dir,
        batch_size=args.batch_size,
        seed=42,
        prefetch=args.prefetch,
    )
    
    model = PhyloLM(
        num_rows=train_iter.num_rows,
        num_cols=train_iter.num_cols,
        num_blocks=args.num_blocks,
        h_dim=args.h_dim,
        num_heads=args.num_heads,
        vocab_size=len(tokenizer),
    ).to(device).to(torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()  #torch.nn.L1Loss()
    train_samples = iter(train_iter)
    
    model = torch.compile(model)

    print("Pre-warming for torch.compile (kernel fusion)...")
    for _ in range(100):
        alignment, distances = next(train_samples)
        alignment = alignment.to(device, non_blocking=True)
        distances = distances.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(alignment)
            loss = criterion(preds, distances)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    print("Starting Profile (2 wait, 5 warmup, 10 active steps)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=5, active=10, repeat=1),
        on_trace_ready=tensorboard_trace_handler('./log/h100_debug'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        
        for step in range(20):
            with record_function("01_data_fetching_cpu"):
                alignment, distances = next(train_samples)
            
            with record_function("02_host_to_device_transfer"):
                alignment = alignment.to(device, non_blocking=True)
                distances = distances.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with record_function("03_forward_pass"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(alignment)
                    loss = criterion(preds, distances)

            with record_function("04_backward_pass"):
                loss.backward()

            with record_function("05_optimizer_step"):
                optimizer.step()

            prof.step()
            if step % 5 == 0:
                print(f"Step {step} complete...")

    print("\n--- Top 15 CPU Functions (Identifying Bottlenecks) ---")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    
    print("\n--- Top 15 CUDA Functions (Identifying Inefficient Kernels) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap_dir", type=str, default="/mloscratch/homes/navasard/protein_stuff/LG_GC_memmaps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--h_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    
    args = parser.parse_args()
    run_profile(args)





































































# import torch
# from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
# from model.data import Tokenizer
# from model.memmap_data import create_memmap_dataloaders
# from model.model import PhyloLM

# def run_profile(args):
#     device = torch.device("cuda")
#     tokenizer = Tokenizer()
    
#     # 1. Profile Data Loading Setup
#     print("Initializing Dataloaders...")
#     train_iter, _ = create_memmap_dataloaders(
#         memmap_dir=args.memmap_dir,
#         batch_size=args.batch_size,
#         seed=42,
#         prefetch=args.prefetch,
#     )
    
#     model = PhyloLM(
#         num_rows=train_iter.num_rows,
#         num_cols=train_iter.num_cols,
#         num_blocks=args.num_blocks,
#         h_dim=args.h_dim,
#         num_heads=args.num_heads,
#         vocab_size=len(tokenizer),
#     ).to(device).to(torch.bfloat16)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#     criterion = torch.nn.L1Loss()
#     train_samples = iter(train_iter)
    
#     model = torch.compile(model)

#     print("Starting Profile (5 warmup steps, 10 active steps)...")
    
#     # The Profiler Configuration
#     with profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=2, warmup=5, active=10, repeat=1),
#         on_trace_ready=tensorboard_trace_handler('./log/h100_debug'),
#         record_shapes=True,
#         with_stack=True,
#         profile_memory=True
#     ) as prof:
        
#         for step in range(20):
#             # --- STAGE 1: DATA FETCHING ---
#             with record_function("01_data_fetching_cpu"):
#                 alignment, distances = next(train_samples)
            
#             # --- STAGE 2: HOST TO DEVICE ---
#             with record_function("02_host_to_device_transfer"):
#                 alignment = alignment.to(device, non_blocking=True)
#                 distances = distances.to(device, non_blocking=True)

#             # --- STAGE 3: FORWARD PASS ---
#             optimizer.zero_grad(set_to_none=True)
#             with record_function("03_forward_pass"):
#                 with torch.autocast("cuda", dtype=torch.bfloat16):
#                     preds = model(alignment)
#                     loss = criterion(preds, distances)

#             # --- STAGE 4: BACKWARD PASS ---
#             with record_function("04_backward_pass"):
#                 loss.backward()

#             # --- STAGE 5: OPTIMIZER ---
#             with record_function("05_optimizer_step"):
#                 optimizer.step()

#             prof.step() # Notify profiler of step end
#             if step % 5 == 0:
#                 print(f"Step {step} complete...")

#     # Output the results
#     print("\n--- Top 15 CPU Functions (Identifying Bottlenecks) ---")
#     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    
#     print("\n--- Top 15 CUDA Functions (Identifying Inefficient Kernels) ---")
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--memmap_dir", type=str, default="/mloscratch/homes/navasard/protein_stuff/LG_GC_memmaps")
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--prefetch", type=int, default=2)
#     parser.add_argument("--num_blocks", type=int, default=6)
#     parser.add_argument("--h_dim", type=int, default=64)
#     parser.add_argument("--num_heads", type=int, default=4)
    
#     args = parser.parse_args()
#     run_profile(args)