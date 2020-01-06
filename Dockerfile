FROM nvcr.io/nvidia/pytorch:19.06-py3

MAINTAINER Deepglint


RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir \
    easydict==1.9 \
    pytorch-ignite==0.2.1 \
    scikit-learn==0.21.0 \
    yacs==0.1.6

RUN rm -fr /tmp/* /var/cache/apt/* && apt-get clean
WORKDIR /workspace

RUN ["/bin/bash"]