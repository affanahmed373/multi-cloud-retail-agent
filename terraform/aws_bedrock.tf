# ---------------------------------------------------------------------------
# AWS — IAM for Bedrock (Frankfurt / eu-central-1)
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {
  count = var.enable_aws ? 1 : 0
}

resource "aws_iam_role" "bedrock_runtime" {
  count = var.enable_aws ? 1 : 0

  name = "${local.name_prefix}-bedrock-runtime"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = [
            "ec2.amazonaws.com",
            "ecs-tasks.amazonaws.com",
            "build.apprunner.amazonaws.com",
            "tasks.apprunner.amazonaws.com",
          ]
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "bedrock_invoke" {
  count = var.enable_aws ? 1 : 0

  name = "${local.name_prefix}-bedrock-invoke"
  role = aws_iam_role.bedrock_runtime[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeBedrockModels"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:ListFoundationModels",
          "bedrock:GetFoundationModel",
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_user" "bedrock_local_dev" {
  count = var.enable_aws ? 1 : 0

  name = "${local.name_prefix}-bedrock-dev"
  path = "/retail-agent/"
}

resource "aws_iam_user_policy" "bedrock_local_dev" {
  count = var.enable_aws ? 1 : 0

  name = "${local.name_prefix}-bedrock-dev-invoke"
  user = aws_iam_user.bedrock_local_dev[0].name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeBedrockModels"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:ListFoundationModels",
          "bedrock:GetFoundationModel",
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "agent" {
  count = var.enable_aws ? 1 : 0

  name              = "/retail-agent/${var.environment}"
  retention_in_days = 14
}
