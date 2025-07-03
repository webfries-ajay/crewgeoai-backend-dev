# Database Migrations

This directory contains Alembic database migration files for the CrewGeoAI backend.

## Setup

1. Install dependencies including Alembic:
```bash
pip install -r requirements.txt
```

2. Set up your database connection in `.env`:
```env
DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/crew
```

## Creating Your First Migration

1. **Initialize the database (if not done already):**
```bash
# This creates the initial migration based on your models
alembic revision --autogenerate -m "Initial migration - user and auth tables"
```

2. **Apply the migration:**
```bash
alembic upgrade head
```

## Common Migration Commands

### Create a new migration
```bash
# Auto-generate migration based on model changes
alembic revision --autogenerate -m "Description of changes"

# Create empty migration file for manual changes
alembic revision -m "Manual migration description"
```

### Apply migrations
```bash
# Apply all pending migrations
alembic upgrade head

# Apply migrations up to a specific revision
alembic upgrade [revision_id]

# Apply only the next migration
alembic upgrade +1
```

### Rollback migrations
```bash
# Rollback to previous revision
alembic downgrade -1

# Rollback to specific revision
alembic downgrade [revision_id]

# Rollback all migrations
alembic downgrade base
```

### Check migration status
```bash
# Show current revision
alembic current

# Show migration history
alembic history

# Show pending migrations
alembic show [revision_id]
```

## Migration Workflow

1. **Make changes to your models** in `models/*.py`
2. **Generate migration**: `alembic revision --autogenerate -m "Description"`
3. **Review the generated migration** in `migrations/versions/`
4. **Apply the migration**: `alembic upgrade head`

## Important Notes

- Always review auto-generated migrations before applying them
- Test migrations on a copy of production data
- Back up your database before running migrations in production
- Never edit migration files that have already been applied
- Use descriptive messages for your migrations

## Example Migration Workflow

```bash
# 1. Create initial migration
alembic revision --autogenerate -m "Initial migration - user and auth tables"

# 2. Apply migration
alembic upgrade head

# 3. Add new model or modify existing
# (Edit your models in models/*.py)

# 4. Generate new migration
alembic revision --autogenerate -m "Add project model"

# 5. Apply new migration
alembic upgrade head
```

## Production Considerations

- Always backup database before migrations
- Run migrations during maintenance windows
- Test migrations on staging environment first
- Monitor database performance after schema changes
- Keep migration files in version control 