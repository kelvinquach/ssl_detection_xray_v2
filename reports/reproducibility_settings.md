# Phase 0 Reproducibility Settings

## Environment

- Python executable: C:\Users\USER\anaconda3\envs\sslxray\python.exe
- Python version: 3.10.20
- Platform: Windows-10-10.0.26200-SP0
- Conda environment: sslxray

## Dependency status

- pip check: No broken requirements found.
- Core imports: PASS
- MMDetection stack: FAIL / deferred
- Local environment role: data/script/report validation only
- Local environment is not considered training-ready.

## CUDA

- CUDA available: false
- torch.version.cuda: null
- GPU device count: 0

## Seed

- Global seed: 2026
- PYTHONHASHSEED: 2026
- Python random seed: enabled
- NumPy seed: enabled
- PyTorch CPU seed: enabled
- PyTorch CUDA seed: not applied because CUDA is unavailable

## Deterministic flags

- torch.use_deterministic_algorithms: true
- torch.backends.cudnn.deterministic: true
- torch.backends.cudnn.benchmark: false
- CUBLAS_WORKSPACE_CONFIG: :4096:8

## Framework decision

MMDetection was not installed in the local Windows CPU-only environment.

Training-related framework setup is deferred to a GPU/remote environment.

The local environment may be used for:

- repository validation
- metadata inspection
- annotation checking
- split generation
- COCO-format validation
- report generation

The local environment must not be used for:

- detector training
- checkpoint selection
- SSL pseudo-label training
- final evaluation
- threshold tuning
- model/backbone selection

## Protocol lock

- Primary metric: mAP@0.5:0.95
- Checkpoint selection split: validation only
- Test usage: final evaluation only

Forbidden:

- Do not use test set for threshold tuning.
- Do not use test set for checkpoint selection.
- Do not use test set for model/backbone selection.
- Do not use test set for augmentation/backbone decisions.