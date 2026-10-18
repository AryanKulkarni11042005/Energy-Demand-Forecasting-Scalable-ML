---
title: Energy Demand Forecasting MLOps
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Energy Demand Forecasting — Zero-Cost Cloud Deployment

Production-grade time-series forecasting of next-day regional electricity demand across US balancing authorities using the **EIA API v2**.

## Hugging Face Spaces Deployment
This Space runs directly from `Dockerfile.serving`, utilizing **16 GB RAM and 2 vCPUs completely free**.

### Endpoints
- Web Dashboard: `/`
- Health: `/health`
- Forecast API: `POST /api/predict`
- Drift & Monitoring Dashboard: `/monitoring/report`
