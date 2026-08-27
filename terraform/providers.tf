provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      RegionHint  = "frankfurt-near-duisburg"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}
