# Deep-Learning Lab

Fashion-MNIST deep learning lab (Task 1–8).

## Contents
- Task 1: Tensors, Autograd, Gradient Descent
- Task 2: Reproducible Data Pipeline (Fashion-MNIST)
- Task 3: Training Engine + MLP Baseline
- Task 4: CNN with BatchNorm
- Task 5: Debugging a Broken Training Script
- Task 6: Regularisation, LR Scheduling, Early Stopping
- Task 7: LSTM Sequence Forecasting
- Task 8: ONNX Export + Dockerised FastAPI Serving

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install numpy==1.26.4
    pip install torch==2.3.1+cpu torchvision==0.18.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
    pip install scikit-learn==1.5.1 onnx==1.16.1 onnxruntime==1.18.1 protobuf==4.25.3
