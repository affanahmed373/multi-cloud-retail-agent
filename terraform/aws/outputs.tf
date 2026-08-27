output "aws_region" {
  description = "AWS region in use (Frankfurt)"
  value       = var.aws_region
}

output "bedrock_runtime_role_arn" {
  description = "IAM role ARN for Bedrock runtime access"
  value       = try(aws_iam_role.bedrock_runtime[0].arn, null)
}

output "bedrock_dev_user_name" {
  description = "IAM user for local Bedrock development"
  value       = try(aws_iam_user.bedrock_local_dev[0].name, null)
}

output "apprunner_service_url" {
  description = "AWS App Runner URL (if enabled)"
  value       = try(aws_apprunner_service.agent[0].service_url, null)
}

output "env_hint" {
  description = "Suggested .env values for this stack"
  value = {
    BEDROCK_REGION     = var.aws_region
    BEDROCK_MODEL_ID   = var.bedrock_model_id
    AWS_REGION = var.aws_region
  }
}
