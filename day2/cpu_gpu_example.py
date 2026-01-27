import torch                                                                                                                                        
import time                                                                                                                                         
                                                                                                                                                    
# Matrix size                                                                                                                                       
n = 4096                                                                                                                                            
                                                                                                                                                    
# Create random matrices                                                                                                                            
a_cpu = torch.randn(n, n)                                                                                                                           
b_cpu = torch.randn(n, n)                                                                                                                           
                                                                                                                                                    
# CPU benchmark                                                                                                                                     
start = time.time()                                                                                                                                 
for _ in range(10):                                                                                                                                 
    c_cpu = torch.matmul(a_cpu, b_cpu)                                                                                                              
cpu_time = (time.time() - start) / 10                                                                                                               
print(f"CPU: {cpu_time*1000:.2f} ms per multiplication")                                                                                            
                                                                                                                                                    
# Check if CUDA is available                                                                                                                        
if torch.cuda.is_available():                                                                                                                       
    # Move to GPU                                                                                                                                   
    a_gpu = a_cpu.to('cuda')                                                                                                                        
    b_gpu = b_cpu.to('cuda')                                                                                                                        
                                                                                                                                                    
    # Warmup                                                                                                                                        
    for _ in range(3):                                                                                                                              
        _ = torch.matmul(a_gpu, b_gpu)                                                                                                              
    torch.cuda.synchronize()                                                                                                                        
                                                                                                                                                    
    # GPU benchmark                                                                                                                                 
    start = time.time()                                                                                                                             
    for _ in range(10):                                                                                                                             
        c_gpu = torch.matmul(a_gpu, b_gpu)                                                                                                          
    torch.cuda.synchronize()                                                                                                                        
    gpu_time = (time.time() - start) / 10                                                                                                           
                                                                                                                                                    
    print(f"GPU: {gpu_time*1000:.2f} ms per multiplication")                                                                                        
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")                                                                                                     
else:                                                                                                                                               
    print("CUDA not available. Enable GPU in Colab: Runtime > Change runtime type > GPU")

import torch                                                                                                                                        
import time                                                                                                                                         
from transformers import AutoTokenizer, AutoModel                                                                                                   
                                                                                                                                                    
# Load model and tokenizer                                                                                                                          
model_name = "bert-base-uncased"                                                                                                                    
tokenizer = AutoTokenizer.from_pretrained(model_name)                                                                                               
model = AutoModel.from_pretrained(model_name)                                                                                                       
                                                                                                                                                    
# Sample text                                                                                                                                       
texts = ["This is a test sentence for benchmarking."] * 32  # batch of 32                                                                           
                                                                                                                                                    
# Tokenize                                                                                                                                          
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)                                                                       
                                                                                                                                                    
# CPU benchmark                                                                                                                                     
model_cpu = model.to("cpu")                                                                                                                         
inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}                                                                                            
                                                                                                                                                    
with torch.no_grad():                                                                                                                               
    start = time.time()                                                                                                                             
    for _ in range(10):                                                                                                                             
        _ = model_cpu(**inputs_cpu)                                                                                                                 
    cpu_time = (time.time() - start) / 10                                                                                                           
print(f"CPU: {cpu_time*1000:.2f} ms per forward pass")                                                                                              
                                                                                                                                                    
# GPU benchmark                                                                                                                                     
if torch.cuda.is_available():                                                                                                                       
    model_gpu = model.to("cuda")                                                                                                                    
    inputs_gpu = {k: v.to("cuda") for k, v in inputs.items()}                                                                                       
                                                                                                                                                    
    # Warmup                                                                                                                                        
    with torch.no_grad():                                                                                                                           
        for _ in range(3):                                                                                                                          
            _ = model_gpu(**inputs_gpu)                                                                                                             
        torch.cuda.synchronize()                                                                                                                    
                                                                                                                                                    
    # Benchmark                                                                                                                                     
    with torch.no_grad():                                                                                                                           
        start = time.time()                                                                                                                         
        for _ in range(10):                                                                                                                         
            _ = model_gpu(**inputs_gpu)                                                                                                             
        torch.cuda.synchronize()                                                                                                                    
        gpu_time = (time.time() - start) / 10                                                                                                       
                                                                                                                                                    
    print(f"GPU: {gpu_time*1000:.2f} ms per forward pass")                                                                                          
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")                                                                                                     
else:                                                                                                                                               
    print("CUDA not available. Enable GPU in Colab: Runtime > Change runtime type > GPU") 
