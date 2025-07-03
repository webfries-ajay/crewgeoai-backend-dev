#!/usr/bin/env python3
"""
Database management script for CrewGeoAI backend
Provides convenient commands for database operations and migrations
"""
import asyncio
import sys
import os
import subprocess
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import logging

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import settings
from core.database import Base, engine
from models import user  # Import all models

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.engine = engine
    
    async def create_database(self):
        """Create the database if it doesn't exist"""
        try:
            # Extract database name from URL
            db_name = settings.database_url.split('/')[-1]
            base_url = settings.database_url.rsplit('/', 1)[0]
            
            # Connect without specifying database
            temp_engine = create_async_engine(
                f"{base_url}/postgres",  # Connect to default postgres DB
                echo=False
            )
            
            async with temp_engine.connect() as conn:
                # Check if database exists
                result = await conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": db_name}
                )
                
                if not result.fetchone():
                    # Database doesn't exist, create it
                    await conn.execute(text("COMMIT"))  # End transaction
                    await conn.execute(text(f"CREATE DATABASE {db_name}"))
                    print(f"✅ Database '{db_name}' created successfully")
                else:
                    print(f"ℹ️ Database '{db_name}' already exists")
            
            await temp_engine.dispose()
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            return False
        
        return True
    
    async def drop_database(self):
        """Drop the database (DANGER!)"""
        try:
            db_name = settings.database_url.split('/')[-1]
            base_url = settings.database_url.rsplit('/', 1)[0]
            
            # Ask for confirmation
            confirm = input(f"⚠️ Are you sure you want to DROP database '{db_name}'? (type 'yes' to confirm): ")
            if confirm.lower() != 'yes':
                print("❌ Operation cancelled")
                return False
            
            temp_engine = create_async_engine(
                f"{base_url}/postgres",
                echo=False
            )
            
            async with temp_engine.connect() as conn:
                # Terminate active connections
                await conn.execute(text("COMMIT"))
                await conn.execute(
                    text(f"""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = :db_name AND pid <> pg_backend_pid()
                    """),
                    {"db_name": db_name}
                )
                
                # Drop database
                await conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
                print(f"✅ Database '{db_name}' dropped successfully")
            
            await temp_engine.dispose()
            
        except Exception as e:
            print(f"❌ Error dropping database: {e}")
            return False
        
        return True
    
    async def init_tables(self):
        """Initialize database tables directly (without migrations)"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            print("✅ Database tables created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return False
    
    async def drop_tables(self):
        """Drop all tables (DANGER!)"""
        try:
            confirm = input("⚠️ Are you sure you want to DROP ALL TABLES? (type 'yes' to confirm): ")
            if confirm.lower() != 'yes':
                print("❌ Operation cancelled")
                return False
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            print("✅ All tables dropped successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error dropping tables: {e}")
            return False
    
    async def check_connection(self):
        """Check database connection"""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
            
            print("✅ Database connection successful")
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    async def show_tables(self):
        """Show all tables in the database"""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(
                    text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                    """)
                )
                
                tables = result.fetchall()
                
                if tables:
                    print("📊 Database tables:")
                    for table in tables:
                        print(f"  - {table[0]}")
                else:
                    print("ℹ️ No tables found in database")
                
                return True
                
        except Exception as e:
            print(f"❌ Error listing tables: {e}")
            return False

def run_alembic_command(command: str):
    """Run Alembic command"""
    try:
        result = subprocess.run(
            f"alembic {command}",
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Alembic command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running Alembic command: {e}")
        return False

async def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("""
🗄️ CrewGeoAI Database Manager

Usage: python manage_db.py <command>

Commands:
  check             - Check database connection
  create-db         - Create database if it doesn't exist
  drop-db           - Drop database (DANGER!)
  
  init              - Initialize tables directly (no migrations)
  drop-tables       - Drop all tables (DANGER!)
  show-tables       - Show all tables in database
  
  # Migration commands (using Alembic)
  init-migrations   - Initialize Alembic migrations
  create-migration  - Create new migration (auto-generate)
  migrate           - Apply all pending migrations
  downgrade         - Rollback last migration
  current           - Show current migration
  history           - Show migration history
  
Examples:
  python manage_db.py check
  python manage_db.py create-db
  python manage_db.py init-migrations
  python manage_db.py create-migration "Add user tables"
  python manage_db.py migrate
        """)
        return
    
    command = sys.argv[1].lower()
    db_manager = DatabaseManager()
    
    # Direct database operations
    if command == "check":
        await db_manager.check_connection()
    
    elif command == "create-db":
        await db_manager.create_database()
    
    elif command == "drop-db":
        await db_manager.drop_database()
    
    elif command == "init":
        await db_manager.init_tables()
    
    elif command == "drop-tables":
        await db_manager.drop_tables()
    
    elif command == "show-tables":
        await db_manager.show_tables()
    
    # Migration commands
    elif command == "init-migrations":
        print("🔄 Initializing Alembic migrations...")
        if run_alembic_command("init migrations"):
            print("✅ Migrations initialized")
        else:
            print("❌ Failed to initialize migrations")
    
    elif command == "create-migration":
        message = input("Enter migration message: ") or "Auto migration"
        print(f"🔄 Creating migration: {message}")
        if run_alembic_command(f'revision --autogenerate -m "{message}"'):
            print("✅ Migration created successfully")
        else:
            print("❌ Failed to create migration")
    
    elif command == "migrate":
        print("🔄 Applying migrations...")
        if run_alembic_command("upgrade head"):
            print("✅ Migrations applied successfully")
        else:
            print("❌ Failed to apply migrations")
    
    elif command == "downgrade":
        print("🔄 Rolling back last migration...")
        if run_alembic_command("downgrade -1"):
            print("✅ Migration rolled back")
        else:
            print("❌ Failed to rollback migration")
    
    elif command == "current":
        print("📍 Current migration:")
        run_alembic_command("current")
    
    elif command == "history":
        print("📚 Migration history:")
        run_alembic_command("history")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python manage_db.py' without arguments to see available commands")

if __name__ == "__main__":
    asyncio.run(main()) 