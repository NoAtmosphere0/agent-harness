---
id: runbook-payment-gateway-outage
title: "Runbook: payment-gateway outage"
services: [payment-gateway, checkout-api]
tags: [runbook, payments, outage]
---
# Runbook: payment-gateway outage

Symptoms: payment-gateway status is outage or its error rate is above 50%; checkout-api shows elevated 5xx errors and slow responses because it waits on payment authorisation.

Impact: customers cannot complete purchases. This is revenue-critical; see the severity matrix (normally SEV1).

Mitigation steps:
1. Confirm with get_service_status on payment-gateway and checkout-api.
2. Check the card processor vendor status page for an upstream incident.
3. Enable the checkout "payment retry later" banner to reduce repeated attempts.
4. If the outage lasts more than 15 minutes, fail over to the secondary processor region.

Escalation: payments on-call, then the payments engineering manager.
