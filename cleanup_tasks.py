#!/usr/bin/env python3
"""
Direct task cleanup - directly modifies SQLite to mark all pending/active/running tasks as 'invalidated'
"""
import sqlite3
import sys

db_path = r'E:\ML PROJECTS\AGENT_ZERO\core_logic\tasks.db'

try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get orphaned tasks
    cur.execute("SELECT id, goal, state FROM tasks WHERE state IN ('pending', 'active', 'running', 'paused')")
    orphaned = cur.fetchall()
    
    print(f"\n=== ORPHANED TASKS CLEANUP ===")
    print(f"Found {len(orphaned)} orphaned tasks:\n")
    
    if orphaned:
        for i, (task_id, goal, state) in enumerate(orphaned, 1):
            goal_preview = goal[:60] if goal else "(no goal)"
            print(f"{i}. {task_id[:8]} | {state:8s} | {goal_preview}")
        
        # Mark them as invalidated instead of deleting
        for task_id, _, _ in orphaned:
            cur.execute("UPDATE tasks SET state = 'invalidated' WHERE id = ?", (task_id,))
        
        conn.commit()
        print(f"\n[OK] Marked {len(orphaned)} tasks as 'invalidated' and cleared from dispatch.")
    else:
        print("[OK] No orphaned tasks found.")
    
    conn.close()
    
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

print("Done.\n")
