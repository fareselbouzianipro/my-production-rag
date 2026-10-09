# Production Deployment Guide

## Deployment process

The Acme API is deployed using Docker containers.

Before deploying, engineers must ensure that all automated tests pass on the main branch.

Deployments are performed through the CI/CD pipeline. Engineers must not deploy manually from their local machines.

## Rollback

If a deployment causes a production incident, the on-call engineer can roll back to the previous stable version using the deployment dashboard.

The incident must be documented after service has been restored.
