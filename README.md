# ForgeML

**ForgeML is our attempt to make experimenting with machine-learning models easier to manage—from fine-tuning and inference to tracking experiments and evaluating results.**

We wanted a single workspace where we could work with models, datasets, training jobs, and outputs instead of juggling disconnected scripts and tools.

## Our inspiration

We were inspired by the growing open-source AI ecosystem and the idea that developers should be able to experiment with models, fine-tune them for specific tasks, and understand their results without needing to build a new workflow from scratch every time.

ForgeML started as a hands-on learning project: build the platform ourselves, test the ML workflow locally first, then use real GPU infrastructure when we are ready.

## What we're building

ForgeML combines a **Next.js dashboard** with a **FastAPI backend**. The project is being developed around:

- Managing models and datasets
- Creating and tracking training jobs
- Experimenting with LoRA fine-tuning
- Running model inference and recording benchmarks
- Organizing artifacts and evaluation results
- Showing experiment history in one dashboard

It is an evolving prototype, and not every workflow is production-ready yet.

## What we've achieved so far

- Built the initial ForgeML dashboard and connected it to FastAPI endpoints.
- Added workspace areas for models, datasets, training jobs, artifacts, evaluations, inference history, and experiments.
- Ran small Qwen2.5 LoRA and inference experiments and saved their configuration and results.
- Recorded a small LoRA smoke test: 10 examples, 2 epochs, batch size 2, 10 steps, and final training loss of 3.435. This was an early technical test, **not proof of model quality or generalization**.
- Recorded inference benchmarks of approximately 58.5 tokens/second for Qwen2.5-3B-Instruct and 65.11 tokens/second for a Qwen2.5-1.5B run. These are results from specific runs, not guaranteed performance figures.
- Set up access to an AMD Developer Cloud GPU droplet for GPU-based experimentation.

## AMD GPU journey

We received **$100 in AMD Developer Cloud credits** through the AMD AI Developer Program. The credit gives us an opportunity to explore GPU computing without paying the full compute cost upfront. AMD describes the program as providing $100 in cloud credits, with the AMD Developer Cloud option offering direct access to AMD Instinct GPUs for training, fine-tuning, and inference. Credit expiry and activation conditions depend on the program terms and activation date.

We provisioned a GPU droplet whose dashboard identifies an **AMD Instinct MI300X configuration with 192 GB of GPU memory**, running **ROCm 10.1.0 on Ubuntu 24.04**. The provider dashboard displayed an estimated cost of **$1.990/hour at the time of the screenshot**; this is a captured estimate, not a promise of current pricing.

Our plan is to be careful with the credits: validate the training and inference pipeline locally where possible, then use the GPU droplet for workloads that genuinely need it. Our goal is to learn the full workflow—model loading, LoRA configuration, training, telemetry, saving adapters, inference, and evaluation—rather than burn compute on unverified experiments.


## Tech stack

- **Frontend:** Next.js, React, TypeScript
- **Backend:** Python, FastAPI, Uvicorn
- **ML experiments:** Hugging Face models, LoRA fine-tuning, inference benchmarks
- **GPU environment:** AMD Instinct MI300X, ROCm, Ubuntu

**Note:** ForgeML is a work in progress. Experiment metrics above are saved results from specific runs, not live measurements. The AMD droplet screenshot documents the provider dashboard at the time it was captured.

**AMD credit reference:** [AMD AI Developer Program — Member Perks](https://developer.amd.com/member-perks/)
