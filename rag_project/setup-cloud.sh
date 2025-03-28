#!/bin/bash
set -e

# Set these variables
PROJECT_ID="rag-application-455020"
REGION="us-central1"
BUCKET_NAME="rag-application-455020-rag-vector-store"

# Create GCP resources
echo "Creating Cloud Storage bucket..."
gsutil mb -l $REGION gs://$BUCKET_NAME || echo "Bucket already exists"

echo "Creating Secret Manager secret for OpenAI API key..."
read -sp "Enter your OpenAI API key: " OPENAI_API_KEY
echo

# Create the secret (will error if it already exists, which is fine)
gcloud secrets create openai-api-key --project=$PROJECT_ID || true
echo -n "$OPENAI_API_KEY" | gcloud secrets versions add openai-api-key --data-file=- --project=$PROJECT_ID

# Since Docker isn't installed locally, use Cloud Build instead
echo "Building using Cloud Build..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/rag-api .

echo "Deploying to Cloud Run..."
gcloud run deploy rag-api \
  --image gcr.io/$PROJECT_ID/rag-api \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars="STORAGE_BUCKET=$BUCKET_NAME,MODEL_NAME=gpt-3.5-turbo,USE_COMPRESSION=true,RETRIEVAL_TOP_K=5,TEMPERATURE=0.1" \
  --set-secrets="OPENAI_API_KEY=openai-api-key:latest" \
  --project=$PROJECT_ID

echo "Deployment complete! Your RAG API is now running on Cloud Run."