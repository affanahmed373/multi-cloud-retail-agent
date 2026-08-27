# ---------------------------------------------------------------------------
# GCP — Vertex AI APIs + service account (Frankfurt / europe-west3)
# ---------------------------------------------------------------------------

resource "google_project_service" "required" {
  for_each = var.enable_gcp ? toset([
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
  ]) : toset([])

  project            = var.gcp_project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_service_account" "vertex_runtime" {
  count = var.enable_gcp ? 1 : 0

  account_id   = "${var.project_name}-${var.environment}-vertex"
  display_name = "Retail Agent Vertex AI (${var.environment})"
  description  = "Runtime identity for Vertex / Gemini calls from the retail agent"
  project      = var.gcp_project_id

  depends_on = [google_project_service.required]
}

resource "google_project_iam_member" "vertex_user" {
  count = var.enable_gcp ? 1 : 0

  project = var.gcp_project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_runtime[0].email}"
}

resource "google_project_iam_member" "vertex_viewer" {
  count = var.enable_gcp ? 1 : 0

  project = var.gcp_project_id
  role    = "roles/aiplatform.viewer"
  member  = "serviceAccount:${google_service_account.vertex_runtime[0].email}"
}

check "gcp_project_required" {
  assert {
    condition     = !var.enable_gcp || var.gcp_project_id != ""
    error_message = "gcp_project_id must be set when enable_gcp = true."
  }
}
