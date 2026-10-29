"""Leave shutdown time inside both the provider and reservation deadlines."""


def worker_budget(ticket, started_at, now, shutdown_seconds=180):
    elapsed = max(0, now - started_at)
    return min(ticket["seconds"] - elapsed, ticket["deadline"] - now) - shutdown_seconds
