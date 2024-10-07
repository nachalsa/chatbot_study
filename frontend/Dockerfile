# syntax=docker/dockerfile:1

# Use an official Node.js image as the base image
FROM node:14

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Copy environment files (assuming they are in the root of the project)
COPY .env.* ./

ARG REACT_APP_URL

ENV REACT_APP_URL $REACT_APP_URL

# # Expose the port that the app runs on
# EXPOSE 3000

# # Build the React application
# ARG REACT_APP_ENV
# RUN if [ "$REACT_APP_ENV" = "production" ]; then \
#     cp .env.production .env; \
#     else cp .env.development .env; \
#     fi && npm run build

# Expose the port that the app runs on
EXPOSE 3000

# Start the React application
CMD ["npm", "start"]
