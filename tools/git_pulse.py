#!/usr/bin/env python3

import requests
from datetime import datetime, timedelta
import argparse
import json
import sys


def get_monday(offset=0):
    today = datetime.utcnow()
    days_ahead = today.weekday()
    monday = today - timedelta(days=days_ahead + offset * 7)
    return monday.isoformat() + 'Z'


def get_commits(repo, since):
    url = f"https://api.github.com/repos/{repo}/commits?since={since}&per_page=100"
    headers = {'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'git_pulse'}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 403:
        print('Rate limited. Wait 60s or use token.')
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def get_prs(repo, since):
    url = f"https://api.github.com/repos/{repo}/pulls?state=all&since={since}&per_page=100&sort=updated&direction=desc"
    headers = {'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'git_pulse'}
    resp = requests.get(url, headers=headers)
    if resp.status_code == 403:
        print('Rate limited. Wait 60s or use token.')
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def summarize_commits(commits):
    authors = {}
    for c in commits:
        author = c['commit']['author']['name']
        authors[author] = authors.get(author, 0) + 1
    return len(commits), authors


def summarize_prs(prs):
    states = {'open': 0, 'closed': 0, 'merged': 0}
    authors = {}
    for pr in prs:
        state = pr['state']
        if state == 'closed' and pr.get('merged_at'):
            states['merged'] += 1
        else:
            states[state] += 1
        author = pr['user']['login']
        authors[author] = authors.get(author, 0) + 1
    return states, len(prs), authors


def main(offset=0):
    repo = 'agent0ai/agent-zero'
    since = get_monday(offset)
    monday_str = datetime.fromisoformat(since[:-1]).strftime('%Y-%m-%d')
    print(f"Git Pulse: {repo} for week starting {monday_str} (UTC)")
    print('=' * 60)

    commits = get_commits(repo, since)
    num_commits, commit_authors = summarize_commits(commits)

    prs = get_prs(repo, since)
    pr_states, num_prs, pr_authors = summarize_prs(prs)

    print(f"\nCommits: {num_commits}")
    for author, count in sorted(commit_authors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {author}: {count}")

    print(f"\nPRs: {num_prs} total")
    print(f"  Open: {pr_states['open']}, Closed: {pr_states['closed']}, Merged: {pr_states['merged']}")
    for author, count in sorted(pr_authors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {author}: {count}")

    velocity = num_commits + num_prs
    print(f"\nSeptember Mode Velocity: {velocity} actions")
    if velocity >= 20:
        print("✅ Hitting goals: High velocity!")
    elif velocity >= 10:
        print("⚡ Steady progress.")
    else:
        print("🚀 Room to accelerate.")

    # Recent 5 commits
    print("\nRecent Commits:")
    for c in commits[:5]:
        msg = c['commit']['message'].split('\n')[0]
        print(f"  {c['sha'][:8]} ({c['commit']['author']['name']}): {msg}")

    # Recent 5 PRs
    print("\nRecent PRs:")
    for pr in prs[:5]:
        state = pr['state'][:1].upper()
        print(f"  #{pr['number']} {state}: {pr['title'][:60]} ({pr['user']['login']})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GitHub repo pulse checker for agent0ai/agent-zero")
    parser.add_argument('--week', '-w', type=int, default=0, help='Week offset from current (0= this week, 1=last, etc.)')
    args = parser.parse_args()
    main(args.week)
