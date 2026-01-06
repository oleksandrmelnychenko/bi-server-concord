# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Build arguments for API URLs (baked into the build)
ARG VITE_RAG_API=http://localhost:8000
ARG VITE_ANALYTICS_API=http://localhost:8001
ARG VITE_SQL_API=http://localhost:8002

# Set as environment variables for build
ENV VITE_RAG_API=$VITE_RAG_API
ENV VITE_ANALYTICS_API=$VITE_ANALYTICS_API
ENV VITE_SQL_API=$VITE_SQL_API

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
