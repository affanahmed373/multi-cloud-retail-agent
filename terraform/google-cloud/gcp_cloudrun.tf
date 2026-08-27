# ---------------------------------------------------------------------------
# GCP Cloud Run — optional FastAPI deploy in Frankfurt
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "agent" {
  count = var.enable_gcp && var.enable_gcp_cloudrun ? 1 : 0

  name     = "${local.name_prefix}-api"
  location = var.gcp_region
  project  = var.gcp_project_id
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.vertex_runtime[0].email

    containers {
      image = var.container_image

      ports {
        container_port = var.container_port
      }

      env {
        name  = "LLM_PROVIDER"
        value = var.llm_provider == "bedrock" ? "vertex" : var.llm_provider
      }

      env {
        name  = "VERTEX_PROJECT_ID"
        value = var.gcp_project_id
      }

      env {
        name  = "VERTEX_LOCATION"
        value = var.gcp_region
      }

      env {
        name  = "VERTEX_MODEL_NAME"
        value = var.vertex_model_name
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  labels = local.common_labels

  depends_on = [google_project_service.required]

  lifecycle {
    precondition {
      condition     = var.container_image != ""
      error_message = "container_image must be set when enable_gcp_cloudrun = true."
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  count = var.enable_gcp && var.enable_gcp_cloudrun ? 1 : 0

  project  = var.gcp_project_id
  location = google_cloud_run_v2_service.agent[0].location
  name     = google_cloud_run_v2_service.agent[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
