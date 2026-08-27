# Google Cloud deployment

Build and publish the container to Artifact Registry, then set `container_image` and enable Cloud Run in `terraform/google-cloud/terraform.tfvars`.

```bash
docker build -t retail-agent .
terraform -chdir=terraform/google-cloud init
terraform -chdir=terraform/google-cloud apply
```