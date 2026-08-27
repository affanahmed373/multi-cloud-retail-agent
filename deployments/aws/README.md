# AWS deployment

Build and publish the container, then set `container_image` and enable App Runner in `terraform/aws/terraform.tfvars`.

```bash
docker build -t retail-agent .
terraform -chdir=terraform/aws init
terraform -chdir=terraform/aws apply
```