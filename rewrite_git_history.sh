#!/bin/bash

# Git History Rewrite Script
# This script rewrites ALL commit history to change authorship
# WARNING: This permanently modifies Git history - backup first!

set -e  # Exit on any error

# Configuration - Using your system Git settings
NEW_NAME="tanaykapoor"
NEW_EMAIL="tanay_kapoor@icloud.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}⚠️  WARNING: This script will permanently rewrite Git history!${NC}"
echo -e "${YELLOW}📋 Current configuration:${NC}"
echo -e "   New Author: ${NEW_NAME} <${NEW_EMAIL}>"
echo ""

# Safety checks
if [[ "$NEW_NAME" == "Your Name" || "$NEW_EMAIL" == "your.email@example.com" ]]; then
    echo -e "${RED}❌ Error: Please update NEW_NAME and NEW_EMAIL variables in the script${NC}"
    exit 1
fi

if [[ ! -d ".git" ]]; then
    echo -e "${RED}❌ Error: Not in a Git repository${NC}"
    exit 1
fi

# Check if repository is clean
if [[ -n $(git status --porcelain) ]]; then
    echo -e "${RED}❌ Error: Repository has uncommitted changes. Please commit or stash them first.${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}📍 Current branch: ${CURRENT_BRANCH}${NC}"

# Backup recommendation
echo -e "${YELLOW}💾 Backup recommendation:${NC}"
echo "   git clone . ../backup-$(basename $(pwd))-$(date +%Y%m%d-%H%M%S)"
echo ""

# Confirmation
read -p "Do you want to proceed with rewriting history? (type 'yes' to continue): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo -e "${YELLOW}Operation cancelled.${NC}"
    exit 0
fi

echo -e "${GREEN}🚀 Starting Git history rewrite...${NC}"

# Create the filter script
FILTER_SCRIPT=$(cat << 'EOF'
if [ "$GIT_COMMIT" = "" ]; then
    # This is for the initial commit case
    export GIT_AUTHOR_NAME="NEW_NAME_PLACEHOLDER"
    export GIT_AUTHOR_EMAIL="NEW_EMAIL_PLACEHOLDER"
    export GIT_COMMITTER_NAME="NEW_NAME_PLACEHOLDER"
    export GIT_COMMITTER_EMAIL="NEW_EMAIL_PLACEHOLDER"
else
    # For all other commits
    export GIT_AUTHOR_NAME="NEW_NAME_PLACEHOLDER"
    export GIT_AUTHOR_EMAIL="NEW_EMAIL_PLACEHOLDER"
    export GIT_COMMITTER_NAME="NEW_NAME_PLACEHOLDER"
    export GIT_COMMITTER_EMAIL="NEW_EMAIL_PLACEHOLDER"
fi
EOF
)

# Replace placeholders in the filter script
FILTER_SCRIPT="${FILTER_SCRIPT//NEW_NAME_PLACEHOLDER/$NEW_NAME}"
FILTER_SCRIPT="${FILTER_SCRIPT//NEW_EMAIL_PLACEHOLDER/$NEW_EMAIL}"

# Execute the filter-branch command
echo -e "${YELLOW}📝 Rewriting commit history...${NC}"

git filter-branch -f --env-filter "$FILTER_SCRIPT" --tag-name-filter cat -- --branches --tags

echo -e "${GREEN}✅ History rewrite completed!${NC}"

# Clean up backup refs
echo -e "${YELLOW}🧹 Cleaning up backup references...${NC}"
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Force garbage collection to remove old objects
echo -e "${YELLOW}🗑️  Running garbage collection...${NC}"
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Show summary
echo -e "${GREEN}📊 Summary:${NC}"
echo "   - All commits now authored by: ${NEW_NAME} <${NEW_EMAIL}>"
echo "   - Commit messages and timestamps preserved"
echo "   - All branches and tags updated"

# Warning about remotes
echo -e "${YELLOW}⚠️  Important notes:${NC}"
echo "   1. If you have remote repositories, you'll need to force push:"
echo "      git push --force-with-lease origin --all"
echo "      git push --force-with-lease origin --tags"
echo ""
echo "   2. Anyone with existing clones will need to re-clone or rebase"
echo ""
echo "   3. This action cannot be easily undone without a backup"

echo -e "${GREEN}🎉 Git history rewrite completed successfully!${NC}"