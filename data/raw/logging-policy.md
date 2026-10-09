# Logging and Data Retention Policy

## Log retention

Application logs are retained for 30 days.

Security audit logs are retained for 180 days.

## Sensitive information

Passwords, API keys, and authentication tokens must never be written to application logs.

When investigating incidents, engineers should use the centralized logging platform rather than copying sensitive log data into external tools.
