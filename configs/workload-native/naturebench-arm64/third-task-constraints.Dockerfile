FROM naturebench-task:s43588-024-00689-2
LABEL workload.owner=naturebench-native-20261102
RUN python -m pip install numpy==1.26.4 protobuf==5.29.6 tensorboard==2.20.0 numdifftools==0.9.41
RUN python -m pip check
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled python -c "import scanpy,scvi,monod; print('Original task scientific imports passed; no model forward')"
