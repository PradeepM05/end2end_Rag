#!/bin/bash

# Stop any running container
echo "Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# Pull latest changes if using Git
# git pull origin main

# Build the new image
echo "Building Docker image..."
docker-compose -f docker-compose.prod.yml build

# Start the container
echo "Starting container..."
docker-compose -f docker-compose.prod.yml up -d

# Check logs
echo "Container started. Showing logs..."
docker-compose -f docker-compose.prod.yml logs -f