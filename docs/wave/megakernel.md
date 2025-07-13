# Wave Megakernel

This document explains Wave's megakernel optimization, which is a transformation in the compilation pipeline that fuses multiple operations into a single kernel to reduce kernel launch overhead and improve overall performance.

## Overview

The megakernel optimization pass combines multiple separate kernel launches into a single unified kernel execution. This transformation addresses several performance bottlenecks:

1. **Kernel Launch Overhead Reduction**
   - Avoid Python wrapper call overhead
   - Avoid tensor conversion overhead
   - Avoid HIP runtime overhead

2. **Avoid Execution Bubbles Between Kernels**
   - Avoid idling compute units while waiting for kernels to fully finish
   - Subsequent kernels can start work immediately as data becomes available, without waiting for previous kernels to fully finish

## How It Works

```python
    @tkw.wave(get_constraints(Phase.PHASE_0))
    def phase_0(
        q: tkl.Memory[S, B, K1, GLOBAL_ADDRESS_SPACE, wave_input_dtype],
        k: tkl.Memory[N_KV, BH, K1, ADDRESS_SPACE, wave_input_dtype],
        v: tkl.Memory[N_KV, BH, N, ADDRESS_SPACE, wave_input_dtype],
        request_indices: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        kv_indices: tkl.Memory[K2, GLOBAL_ADDRESS_SPACE, tkl.i32],
        logits: tkl.Memory[U, S, B, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
      ...

    @tkw.wave(get_constraints(Phase.PHASE_1))
    def phase_1(
        logits: tkl.Memory[U, S, B, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        request_indices: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        output: tkl.Memory[S, B, N, GLOBAL_ADDRESS_SPACE, wave_output_dtype],
    ):
      ...

   # Fuse and compile the kernels
   @tkw.fuse_kernels
   def fused(
        q: tkl.Memory[S, B, K1, GLOBAL_ADDRESS_SPACE, wave_input_dtype],
        k: tkl.Memory[N_KV, BH, K1, ADDRESS_SPACE, wave_input_dtype],
        v: tkl.Memory[N_KV, BH, N, ADDRESS_SPACE, wave_input_dtype],
        request_indices: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        kv_indices: tkl.Memory[K2, GLOBAL_ADDRESS_SPACE, tkl.i32],
        logits: tkl.Memory[U, S, B, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[S, B, N, GLOBAL_ADDRESS_SPACE, wave_output_dtype],
   ):
      phase_0(q, k, v, request_indices, kv_indices, logits, logits_max, options=compile_options1)
      phase_1(logits, logits_max, request_indices, output, options=compile_options2)

   # Call the fused kernel
   fused(q, k, v, request_indices, kv_indices, logits, logits_max, output)

```

The code above demonstrates Wave's megakernel fusion API through a paged attention example.

1. **Fused Megakernel**
   - `fused`: Combines both phases into a single kernel launch using the `@tkw.fuse_kernels` decorator
   - Takes all inputs/outputs needed across both phases
   - Internally compiles `phase_0` and `phase_1` with appropriate compile options
   - As many kernels as needed can be scheduled in such a manner

2. **Execution Flow**
   - Total count of launched blocks is the sum of `phase_0` and `phase_1` blocks
   - Each launched block will take work to process sequentially, starting from the first phase
   - Uses an atomic counter to determine how much work was completed; the second phase will only start after all blocks from the first phase have finished processing

High-level execution logic:
```python
id = get_block_id()
if id < block_count_phase_0:
   phase_0()
   work_counter += 1
else:
   while work_counter < block_count_phase_0:
      wait()

   phase_1()
```

This can also be generalized to an arbitrary number of stages:
```python
id = get_block_id()
if id < block_count_phase_0:
   phase_0()
   work_counter += 1
if id < (block_count_phase_0 + block_count_phase_1):
   while work_counter < block_count_phase_0:
      wait()

   phase_1()
   work_counter += 1
else:
   while work_counter < (block_count_phase_0 + block_count_phase_1):
      wait()

   phase_2()
```

Overall execution flow will look like this:

```mermaid
block-beta
    columns 1
    block
        running
        completed
        waiting
        space:5
    end

    block:batch1["step 1"]
            a1["kernel 1 block 0"]
            a2["kernel 1 block 1"]
            a3["kernel 1 block 2"]
            space
        a1-->a2
        a2-->a3
        a3-->b1
            b1["kernel 2 block 0"]
            b2["kernel 2 block 1"]
            b3["kernel 2 block 2"]
        b1-->b2
        b2-->b3
        b3-->b2
    end
    block:batch2["step 2"]
            a4["kernel 1 block 0"]
            a5["kernel 1 block 1"]
            a6["kernel 1 block 2"]
            space
        a4-->a5
        a5-->a6
        a6-->b4
            b4["kernel 2 block 0"]
            b5["kernel 2 block 1"]
            b6["kernel 2 block 2"]
        b4-->b5
        b5-->b6
    end
    block:batch3["step 3"]
            a7["kernel 1 block 0"]
            a8["kernel 1 block 1"]
            a9["kernel 1 block 2"]
            space
        a7-->a8
        a8-->a9
        a9-->b7
            b7["kernel 2 block 0"]
            b8["kernel 2 block 1"]
            b9["kernel 2 block 2"]
        b7-->b8
        b8-->b9
    end

    style running fill:aqua
    style completed fill:green
    style waiting fill:red
    style a1 fill:aqua
    style a2 fill:aqua
    style a4 fill:green
    style a5 fill:green
    style a6 fill:aqua
    style b4 fill:red
    style a7 fill:green
    style a8 fill:green
    style a9 fill:green
    style b7 fill:aqua
    style b8 fill:aqua
```

Some observations:

1. All kernels need to have the same number of threads
2. There are bubbles in the execution as `phase_1` has to wait for all `phase_0` blocks to complete
3. This approach requires blocks to be scheduled in a sequential manner, which is not guaranteed

### Handling Block Scheduling Order

1. This approach requires blocks to be scheduled in a specific order
2. Block scheduling order is generally not guaranteed; instead of relying on hardware block ID, introduce a virtual block ID
3. Each workgroup will do `fetch_add` on the global atomic counter at the start to get a sequential unique ID
4. Delinearize from 1D to ND block ID if necessary

### Handling Bubbles in Execution Pipeline

Workloads are often scheduled in batches, where each batch's data is independent from other batches.
We can exploit this to reduce the amount of idling the second kernel will need to do.

```python
   # Fuse and compile the kernels
   @tkw.fuse_kernels(batch_dims=[S])
   def fused(
        q: tkl.Memory[S, B, K1, GLOBAL_ADDRESS_SPACE, wave_input_dtype],
        k: tkl.Memory[N_KV, BH, K1, ADDRESS_SPACE, wave_input_dtype],
        v: tkl.Memory[N_KV, BH, N, ADDRESS_SPACE, wave_input_dtype],
        request_indices: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        kv_indices: tkl.Memory[K2, GLOBAL_ADDRESS_SPACE, tkl.i32],
        logits: tkl.Memory[U, S, B, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[S, B, N, GLOBAL_ADDRESS_SPACE, wave_output_dtype],
   ):
      phase_0(q, k, v, request_indices, kv_indices, logits, logits_max, options=compile_options1)
      phase_1(logits, logits_max, request_indices, output, options=compile_options2)
```

Here, we specify our batch dimension `[S]`:

1. Instead of having a single atomic work counter for the kernel, allocate a separate counter for each batch
2. Schedule blocks in a specific Z order to minimize waiting time; see the following execution diagram:

```mermaid
block-beta
block
    columns 1

    block:batch1["batch 0"]
            a1["kernel 1 block 0"]
            a2["kernel 1 block 1"]
            a3["kernel 1 block 2"]
            space
        a1-->a2
        a2-->a3
            b1["kernel 2 block 0"]
            b2["kernel 2 block 1"]
            b3["kernel 2 block 2"]
        b1-->b2
        b2-->b3
        b3-->b2
    end
    block:batch2["batch 1"]
            a4["kernel 1 block 3"]
            a5["kernel 1 block 4"]
            a6["kernel 1 block 5"]
            space
        a3-->a4
        a4-->a5
        a5-->a6
            b4["kernel 2 block 3"]
            b5["kernel 2 block 4"]
            b6["kernel 2 block 5"]
        b3-->b4
        b4-->b5
        b5-->b6
    end
    block:batch3["batch 2"]
            a7["kernel 1 block 6"]
            a8["kernel 1 block 7"]
            a9["kernel 1 block 8"]
            space
        a6-->a7
        a7-->a8
        a8-->a9
        a9-->b1
            b7["kernel 2 block 6"]
            b8["kernel 2 block 7"]
            b9["kernel 2 block 8"]
        b6-->b7
        b7-->b8
        b8-->b9
    end
end
```

Here, if we have enough batches, `kernel 2 block 0` dependencies will most likely be completed by the time execution gets to it.

Kernel logic will look like:
```python
id = get_block_id()
batch_id = get_batch_id(id)
if id < block_count_phase_0:
   phase_0()
   work_counter[batch_id] += 1
else:
   while work_counter[batch_id] < batch_size:
      wait()

   phase_1()
```
