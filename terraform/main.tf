# Anbu Clinic MedAssist — Terraform Infrastructure
# Author: Rajaganapathy M, SRM University | Patent: 202641043947
# Azure Container Apps — Central India
# Version: Day 5 — IaC deployment

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }
  required_version = ">= 1.5.0"
}

provider "azurerm" {
  features {}
}

# ── Resource Group ────────────────────────────────────────────────────────────
resource "azurerm_resource_group" "anbu" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# ── Log Analytics Workspace ───────────────────────────────────────────────────
resource "azurerm_log_analytics_workspace" "anbu" {
  name                = "${var.app_name}-logs"
  location            = azurerm_resource_group.anbu.location
  resource_group_name = azurerm_resource_group.anbu.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

# ── Container Apps Environment ────────────────────────────────────────────────
resource "azurerm_container_app_environment" "anbu" {
  name                       = "${var.app_name}-env"
  location                   = azurerm_resource_group.anbu.location
  resource_group_name        = azurerm_resource_group.anbu.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.anbu.id
  tags                       = var.tags
}

# ── Container App ─────────────────────────────────────────────────────────────
resource "azurerm_container_app" "anbu" {
  name                         = var.app_name
  container_app_environment_id = azurerm_container_app_environment.anbu.id
  resource_group_name          = azurerm_resource_group.anbu.name
  revision_mode                = "Single"
  tags                         = var.tags

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }

    cors_policy {
      allowed_origins  = ["https://rajaganaa.github.io"]
      allowed_methods  = ["GET", "POST", "OPTIONS"]
      allowed_headers  = ["*"]
    }
  }

  secret {
    name  = "groq-api-key"
    value = var.groq_api_key
  }

  secret {
    name  = "github-token"
    value = var.github_token
  }

  secret {
    name  = "wandb-api-key"
    value = var.wandb_api_key
  }

  template {
    min_replicas = 0
    max_replicas = 1

    container {
      name   = var.app_name
      image  = var.container_image
      cpu    = 0.5
      memory = "1Gi"

      env {
        name        = "GROQ_API_KEY"
        secret_name = "groq-api-key"
      }

      env {
        name        = "GITHUB_TOKEN"
        secret_name = "github-token"
      }

      env {
        name        = "WANDB_API_KEY"
        secret_name = "wandb-api-key"
      }

      env {
        name  = "GROQ_MODEL"
        value = "llama-3.3-70b-versatile"
      }

      env {
        name  = "WANDB_PROJECT"
        value = "rrr-clinic-medassist"
      }

      env {
        name  = "WANDB_ENTITY"
        value = "rajaganaa-ai"
      }

      env {
        name  = "PORT"
        value = "8000"
      }

      liveness_probe {
        path             = "/health"
        port             = 8000
        transport        = "HTTP"
        initial_delay    = 30
        interval_seconds = 30
      }

      readiness_probe {
        path             = "/health"
        port             = 8000
        transport        = "HTTP"
        initial_delay    = 10
        interval_seconds = 10
      }
    }
  }
}
