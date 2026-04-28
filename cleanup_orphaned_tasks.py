import sqlite3
import sys

db_path = 'core_logic/tasks.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Check for orphaned tasks
cur.execute("SELECT id, goal, state FROM tasks WHERE state IN ('pending', 'active', 'running')")
orphaned = cur.fetchall()

print(f"\n=== ORPHANED TASKS REPORT ===")
print(f"Total orphaned tasks: {len(orphaned)}\n")

if orphaned:
    for i, (task_id, goal, state) in enumerate(orphaned, 1):
        print(f"{i}. ID: {task_id[:8]} | State: {state} | Goal: {goal[:60]}")
    
    # Delete orphaned tasks
    print(f"\n=== DELETING ORPHANED TASKS ===")
    for task_id, _, _ in orphaned:
        cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    
    conn.commit()
    print(f"Deleted {len(orphaned)} orphaned task(s)")
else:
    print("No orphaned tasks found. Database is clean.")

conn.close()
print("\nDone.")
