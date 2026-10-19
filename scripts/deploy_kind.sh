#!/usr/bin/env bash
# Build the image, create a local kind cluster, and deploy API + ingress + monitoring.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER=heart-mlops
IMAGE=heart-api:1.0.0
INGRESS_NGINX=https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.3/deploy/static/provider/kind/deploy.yaml

echo ">> building $IMAGE"
docker build -t "$IMAGE" -t heart-api:latest .

if ! kind get clusters | grep -qx "$CLUSTER"; then
  echo ">> creating kind cluster $CLUSTER"
  kind create cluster --config k8s/kind-cluster.yaml --wait 120s
fi
kubectl config use-context "kind-$CLUSTER"

echo ">> loading image into the cluster"
kind load docker-image "$IMAGE" --name "$CLUSTER"

echo ">> installing ingress-nginx"
kubectl apply -f "$INGRESS_NGINX"
kubectl wait -n ingress-nginx --for=condition=ready pod \
  -l app.kubernetes.io/component=controller --timeout=180s

echo ">> deploying the API"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml \
              -f k8s/ingress.yaml -f k8s/pdb.yaml
kubectl rollout status -n heart-mlops deployment/heart-api --timeout=180s

echo ">> deploying Prometheus + Grafana"
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl apply -f k8s/monitoring/grafana-configmaps.yaml -f k8s/monitoring/grafana.yaml
kubectl rollout status -n monitoring deployment/prometheus --timeout=180s
kubectl rollout status -n monitoring deployment/grafana --timeout=180s

kubectl get all -n heart-mlops
echo
echo "API (via ingress):  http://localhost:8088/docs"
echo "Prometheus:         http://localhost:9095"
echo "Grafana:            http://localhost:3030"
