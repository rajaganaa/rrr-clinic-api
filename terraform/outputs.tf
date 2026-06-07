# outputs.tf — Useful values after apply

output "app_url" {
  description = "Live URL of Anbu Clinic MedAssist API"
  value       = "https://${azurerm_container_app.anbu.ingress[0].fqdn}"
}

output "resource_group" {
  description = "Resource Group name"
  value       = azurerm_resource_group.anbu.name
}

output "container_app_name" {
  description = "Container App name"
  value       = azurerm_container_app.anbu.name
}

output "environment_name" {
  description = "Container Apps Environment name"
  value       = azurerm_container_app_environment.anbu.name
}

output "log_analytics_workspace" {
  description = "Log Analytics Workspace name"
  value       = azurerm_log_analytics_workspace.anbu.name
}
