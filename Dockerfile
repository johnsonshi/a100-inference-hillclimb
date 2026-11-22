FROM mcr.microsoft.com/azurelinux/base/python:3.12
ARG KUBECTL_VERSION=v1.37.2
RUN tdnf install -y ca-certificates curl tar && tdnf clean all \
 && curl -fsSL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
 && chmod +x /usr/local/bin/kubectl
WORKDIR /app
COPY harness/ /app/harness/
COPY campaigns/ /app/campaigns/
ENV PYTHONUNBUFFERED=1 HILLCLIMB_KUBE_CONTEXT=""
ENTRYPOINT ["python3", "-m", "harness.campaign"]
