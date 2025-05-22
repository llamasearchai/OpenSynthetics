#!/bin/bash

# Exit on error
set -e

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    exit 1
fi

# Load the GitHub token from .env
source .env

# Check if GITHUB_TOKEN is set and not the placeholder
if [ -z "$GITHUB_TOKEN" ] || [ "$GITHUB_TOKEN" = "your_github_token_here" ]; then
    echo "Error: GitHub token not properly configured in .env file!"
    echo "Please edit the .env file and set a valid GITHUB_TOKEN."
    exit 1
fi

# Make sure Git is set up with the correct user
GIT_USER=$(git config user.name)
GIT_EMAIL=$(git config user.email)

if [ -z "$GIT_USER" ] || [ -z "$GIT_EMAIL" ]; then
    echo "Git user not configured. Configuring as Nik Jois..."
    git config user.name "Nik Jois"
    git config user.email "nikjois@llamasearch.ai"
fi

# Make sure we're on the main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Switching to main branch..."
    git checkout main
fi

# Make sure everything is added to git
echo "Checking for any unstaged changes..."
git add -A

# Check if there are any changes to commit
if git diff-index --quiet HEAD --; then
    echo "No changes to commit."
else
    echo "Committing changes..."
    git commit -m "Update documentation and add UI image"
fi

# Set up the remote URL with the token
REMOTE_URL=$(git remote get-url origin)
NEW_REMOTE_URL=$(echo $REMOTE_URL | sed -E "s|https://github.com|https://$GITHUB_TOKEN@github.com|")

echo "Setting up remote with token authentication..."
git remote set-url origin $NEW_REMOTE_URL

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

# Reset the remote URL to remove the token (for security)
echo "Cleaning up..."
git remote set-url origin $REMOTE_URL

echo "Push completed successfully!"
echo "Repository available at: https://github.com/llamasearchai/OpenSynthetics" 