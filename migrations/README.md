# Database migrations

Alembic migrations for the `fraud_tracking` PostgreSQL schema.

Apply all migrations:

```bash
python -m scripts.migrate_database upgrade
```

Show the current revision:

```bash
python -m scripts.migrate_database current
```
