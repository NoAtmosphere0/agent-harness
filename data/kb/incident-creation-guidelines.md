---
id: incident-creation-guidelines
title: Incident creation guidelines
services: []
tags: [incident, process, guidelines]
---
# Incident creation guidelines

Investigate before you open an incident: check the status of the affected service and of its dependencies, and read the relevant runbook.

Open one incident per root cause. If checkout-api is failing because payment-gateway is down, open a single incident for the payment-gateway outage and mention checkout-api as impacted.

Title format: `<service>: <symptom>`, for example `payment-gateway: full outage causing checkout failures`.

The description must contain: customer impact, evidence (status, error rate, p95 latency), suspected cause, and the severity justification quoting the severity matrix threshold that applies.

Do not open incidents for services in an announced maintenance window unless the window has overrun. Incident creation requires human approval; if the approver rejects it, do not retry the same incident.
