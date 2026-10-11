#!/bin/sh
# .agents/skills is the source of truth (read by Codex). Claude Code reads
# .claude/skills, so mirror it. Run after editing any skill, then commit both.
set -e
cd "$(dirname "$0")/.."
rm -rf .claude/skills
mkdir -p .claude
cp -R .agents/skills .claude/skills
echo "synced .agents/skills -> .claude/skills"
