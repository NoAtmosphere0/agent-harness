---
id: service-dependency-map
title: Service dependency map
services: [checkout-api, payment-gateway, auth-service, inventory-service, notification-service, search-api]
tags: [architecture, dependencies]
---
# Service dependency map

checkout-api calls payment-gateway (payment authorisation) and inventory-service (stock reservation). A failure in either shows up as checkout-api errors.

search-api reads from inventory-service for stock levels.

auth-service, payment-gateway, inventory-service and notification-service have no internal dependencies. payment-gateway relies on an external card processor vendor.

When investigating a degraded service, check its dependencies first; the deepest failing dependency is usually the root cause.
