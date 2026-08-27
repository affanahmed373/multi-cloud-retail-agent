# ---------------------------------------------------------------------------
# AWS App Runner — optional FastAPI deploy in Frankfurt
# ---------------------------------------------------------------------------

resource "aws_iam_role" "apprunner_instance" {
  count = var.enable_aws && var.enable_aws_apprunner ? 1 : 0

  name = "${local.name_prefix}-apprunner-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "tasks.apprunner.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_bedrock" {
  count = var.enable_aws && var.enable_aws_apprunner ? 1 : 0

  role       = aws_iam_role.apprunner_instance[0].name
  policy_arn = aws_iam_policy.apprunner_bedrock[0].arn
}

resource "aws_iam_policy" "apprunner_bedrock" {
  count = var.enable_aws && var.enable_aws_apprunner ? 1 : 0

  name = "${local.name_prefix}-apprunner-bedrock"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "${aws_cloudwatch_log_group.agent[0].arn}:*"
      }
    ]
  })
}

resource "aws_iam_role" "apprunner_access" {
  count = var.enable_aws && var.enable_aws_apprunner ? 1 : 0

  name = "${local.name_prefix}-apprunner-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  count = var.enable_aws && var.enable_aws_apprunner ? 1 : 0

  role       = aws_iam_role.apprunner_access[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_apprunner_service" "agent" {
  count = var.enable_aws && var.enable_aws_apprunner ? 1 : 0

  service_name = "${local.name_prefix}-api"

  source_configuration {
    auto_deployments_enabled = false

    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_access[0].arn
    }

    image_repository {
      image_identifier      = var.container_image
      image_repository_type = "ECR"

      image_configuration {
        port = tostring(var.container_port)

        runtime_environment_variables = {
          LLM_PROVIDER     = var.llm_provider
          BEDROCK_REGION   = var.aws_region
          BEDROCK_MODEL_ID = var.bedrock_model_id
        }
      }
    }
  }

  instance_configuration {
    cpu               = "1024"
    memory            = "2048"
    instance_role_arn = aws_iam_role.apprunner_instance[0].arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/health"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  tags = {
    Name = "${local.name_prefix}-api"
  }

  lifecycle {
    precondition {
      condition     = var.container_image != ""
      error_message = "container_image must be set when enable_aws_apprunner = true (e.g. ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/retail-agent:latest)."
    }
  }
}
