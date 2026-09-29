---
id: runbook-checkout-api-5xx
title: "Runbook: checkout-api elevated 5xx errors"
services: [checkout-api, payment-gateway, inventory-service]
tags: [runbook, checkout, errors]
---
# Runbook: checkout-api elevated 5xx errors

Symptoms: checkout-api error rate above 2% or p95 latency above 1500 ms; customers report failed checkouts.

Diagnosis: checkout-api depends on payment-gateway and inventory-service. Check both with get_service_status before blaming checkout-api itself. Most checkout 5xx incidents are caused by a failing dependency.

Mitigation steps:
1. If a dependency is degraded or in outage, follow that dependency's runbook and treat it as the root cause.
2. If both dependencies are healthy, check the latest checkout-api deploy and roll back if it started after the error spike.
3. If database errors appear in logs, see the connection pool exhaustion runbook.

Severity: use the severity matrix; checkout-api is revenue-critical.
