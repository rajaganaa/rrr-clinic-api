# variables.tf — All parameterized inputs

variable "resource_group_name" {
  description = "Azure Resource Group name"
  type        = string
  default     = "antahkarana-rg"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "centralindia"
}

variable "app_name" {
  description = "Container App name"
  type        = string
  default     = "rrr-clinic-api"
}

variable "container_image" {
  description = "Docker image to deploy"
  type        = string
  default     = "ghcr.io/rajaganaa/rrr-clinic-api:latest"
}

variable "groq_api_key" {
  description = "Groq API Key"
  type        = string
  sensitive   = true
}

variable "github_token" {
  description = "GitHub Token for GPT-4o Vision"
  type        = string
  sensitive   = true
}

variable "wandb_api_key" {
  description = "Weights & Biases API Key"
  type        = string
  sensitive   = true
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default = {
    project    = "anbu-clinic-medassist"
    author     = "Rajaganapathy M"
    university = "SRM University"
    patent     = "202641043947"
    env        = "production"
  }
}
