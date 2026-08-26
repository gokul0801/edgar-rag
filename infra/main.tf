# infra/main.tf
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region"            { default = "us-east-2" }
variable "bucket"            { default = "gokul-edgar-rag" }
variable "anthropic_api_key" { sensitive = true }

data "aws_caller_identity" "me" {}

locals {
  image = "${data.aws_caller_identity.me.account_id}.dkr.ecr.${var.region}.amazonaws.com/edgar-rag:latest"
}

resource "aws_iam_role" "lambda" {
  name = "edgar-rag-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda" {
  role = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.bucket}",
          "arn:aws:s3:::${var.bucket}/*",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:${var.region}::foundation-model/amazon.titan-embed-text-v2:0"
      },
    ]
  })
}

resource "aws_lambda_function" "api" {
  function_name = "edgar-rag"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = local.image
  architectures = ["arm64"]
  memory_size   = 2048
  timeout       = 60

  environment {
    variables = {
      DB_PATH           = "s3://${var.bucket}/lancedb"
      TABLE_NAME        = "filings"
      BEDROCK_REGION    = var.region
      AWS_REGION_OVERRIDE = var.region
      ENABLE_RERANK     = "false"
      ANTHROPIC_API_KEY = var.anthropic_api_key
    }
  }
}

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "AWS_IAM"
}

output "url" {
  value = aws_lambda_function_url.api.function_url
}
