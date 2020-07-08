#!/usr/bin/python3

import torch
import apex.amp as amp
import time # for timing execution

# Iterations per test
niter = 100

# Data sizes
N, D_in, D_out = 1024, 8192, 2048

# Results vectors
results_list = []
results_names = []

# Full precision (float32) training
x = torch.randn(N, D_in, device="cuda")
y = torch.randn(N, D_out, device="cuda")
model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Warm up
for t in range(10):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Running full precision...")
tic = time.perf_counter()
for t in range(niter):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
toc = time.perf_counter()
results_list.append(toc-tic)
results_names.append("Float32")
print("Full precision: %0.2f seconds" % (toc-tic))

# Training with AMP
x = torch.randn(N, D_in, device="cuda")
y = torch.randn(N, D_out, device="cuda")
for opt_level in ["O0", "O1", "O2", "O3"]:
    model = torch.nn.Linear(D_in, D_out).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    print("Running AMP with opt level %s" % opt_level)

    # Warm up
    for t in range(10):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

    tic = time.perf_counter()
    for t in range(niter):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
    toc = time.perf_counter()
    results_list.append(toc-tic)
    results_names.append("AMP " + opt_level)
    print("AMP (opt level %s): %0.2f seconds" % (opt_level, toc-tic))

print("\nResults summary (%d iterations)\n===============" % niter)
for name, result in zip(results_names, results_list):
    print("%s: %0.2f seconds  (%0.2fx full precision speed)" % (name, result, results_list[0]/result))
