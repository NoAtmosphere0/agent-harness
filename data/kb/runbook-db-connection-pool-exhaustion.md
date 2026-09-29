---
id: runbook-db-connection-pool-exhaustion
title: "Runbook: database connection pool exhaustion"
services: [checkout-api, inventory-service]
tags: [runbook, database, connection-pool]
---
# Runbook: database connection pool exhaustion

Symptoms: "timeout acquiring connection" errors, requests queueing, p95 latency climbing while database CPU stays low.

Mitigation steps:
1. Identify the service holding connections (active connections per pod on the database dashboard).
2. Restart the pods with the most leaked or idle-in-transaction connections, one at a time.
3. Temporarily raise the pool size by at most 50% if the database has spare capacity.
4. Look for long-running transactions and kill them after confirming with the owning team.

Prevention: set statement and idle-in-transaction timeouts; alert on pool utilisation above 80%.
