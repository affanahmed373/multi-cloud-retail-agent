variable "project_name" {
  description = "Short name used for resource naming"
  type        = string
  default     = "retail-agent"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# ---------------------------------------------------------------------------
# Regions — Frankfurt (nearest major cloud region to Duisburg)
# ---------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region (eu-central-1 = Frankfurt, nearest to Duisburg)"
  type        = string
  default     = "eu-central-1"
}

variable "gcp_region" {
  description = "GCP region (europe-west3 = Frankfurt, nearest to Duisburg)"
  type        = string
  default     = "europe-west3"
}

variable "gcp_project_id" {
  description = "GCP project ID (required when enable_gcp = true)"
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Feature toggles
# ---------------------------------------------------------------------------

variable "enable_aws" {
  description = "Provision AWS Bedrock IAM (+ optional App Runner)"
  type        = bool
  default     = true
}

variable "enable_gcp" {
  description = "Provision GCP Vertex AI IAM (+ optional Cloud Run)"
  type        = bool
  default     = true
}

variable "enable_aws_apprunner" {
  description = "Deploy the FastAPI agent on AWS App Runner"
  type        = bool
  default     = false
}

variable "enable_gcp_cloudrun" {
  description = "Deploy the FastAPI agent on GCP Cloud Run"
  type        = bool
  default     = false
}

# ---------------------------------------------------------------------------
# LLM / app settings
# ---------------------------------------------------------------------------

variable "bedrock_model_id" {
  description = "Default Amazon Bedrock model ID"
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

variable "vertex_model_name" {
  description = "Default Vertex AI / Gemini model name"
  type        = string
  default     = "gemini-1.5-flash"
}

variable "llm_provider" {
  description = "Default LLM_PROVIDER for deployed services (bedrock | vertex | mock)"
  type        = string
  default     = "bedrock"
}

variable "container_image" {
  description = "Container image for App Runner / Cloud Run (must expose port 8000)"
  type        = string
  default     = ""
}

variable "container_port" {
  description = "Container listen port"
  type        = number
  default     = 8000
}
