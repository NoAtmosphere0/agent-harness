---
id: severity-matrix
title: Incident severity matrix
services: [checkout-api, payment-gateway, auth-service, inventory-service, notification-service, search-api]
tags: [severity, incident, triage]
---
# Incident severity matrix

| Severity | Condition (revenue-critical = checkout-api, payment-gateway, auth-service) |
|---|---|
| SEV1 | Revenue-critical service in outage or error rate >= 25%; data loss; security breach |
| SEV2 | Revenue-critical service degraded: error rate 5-25% or p95 > 2000 ms |
| SEV3 | Other customer-facing service degraded, or revenue-critical error rate 1-5% |
| SEV4 | No customer impact |

Pick the highest severity whose condition is met and quote the metric that meets it in the incident description. Planned maintenance inside an announced window is not an incident.

## Response expectations

SEV1: page the on-call engineer and incident commander immediately; status page update within 15 minutes. SEV2: page on-call; update every 30 minutes. SEV3 and SEV4: ticket, handled in business hours.
