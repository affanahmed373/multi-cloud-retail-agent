output "aws_region" {
  description = "AWS region in use (Frankfurt)"
  value       = var.aws_region
}

output "gcp_region" {
  description = "GCP region in use (Frankfurt)"
  value       = var.gcp_region
}

output "bedrock_runtime_role_arn" {
  description = "IAM role ARN for Bedrock runtime access"
  value       = try(aws_iam_role.bedrock_runtime[0].arn, null)
}

output "bedrock_dev_user_name" {
  description = "IAM user for local Bedrock development"
  value       = try(aws_iam_user.bedrock_local_dev[0].name, null)
}

output "vertex_service_account_email" {
  description = "GCP service account for Vertex AI"
  value       = try(google_service_account.vertex_runtime[0].email, null)
}

output "apprunner_service_url" {
  description = "AWS App Runner URL (if enabled)"
  value       = try(aws_apprunner_service.agent[0].service_url, null)
}

output "cloudrun_service_uri" {
  description = "GCP Cloud Run URI (if enabled)"
  value       = try(google_cloud_run_v2_service.agent[0].uri, null)
}

output "env_hint" {
  description = "Suggested .env values for this stack"
  value = {
    BEDROCK_REGION     = var.aws_region
    BEDROCK_MODEL_ID   = var.bedrock_model_id
    VERTEX_LOCATION    = var.gcp_region
    VERTEX_MODEL_NAME  = var.vertex_model_name
    VERTEX_PROJECT_ID  = var.gcp_project_id
  }
}
