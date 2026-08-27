output "gcp_region" {
  description = "GCP region in use (Frankfurt)"
  value       = var.gcp_region
}

output "vertex_service_account_email" {
  description = "GCP service account for Vertex AI"
  value       = try(google_service_account.vertex_runtime[0].email, null)
}

output "cloudrun_service_uri" {
  description = "GCP Cloud Run URI (if enabled)"
  value       = try(google_cloud_run_v2_service.agent[0].uri, null)
}

output "env_hint" {
  description = "Suggested .env values for this stack"
  value = {
    VERTEX_LOCATION   = var.gcp_region
    VERTEX_MODEL_NAME  = var.vertex_model_name
    VERTEX_PROJECT_ID  = var.gcp_project_id
  }
}
