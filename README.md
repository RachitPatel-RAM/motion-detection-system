# Motion Detection System Backend

A Python-based motion detection system with Flask API backend, Firebase integration, and advanced features.

## Features

- Motion detection with contour analysis
- Natural vs unnatural motion filtering
- Night mode adjustments
- Cloudinary for cloud storage
- Gmail for email alerts
- Local storage option
- Video recording with pre-buffer
- Firebase Realtime Database integration
- JWT authentication
- User-specific configurations

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and update with your credentials
6. Run the application: `python main.py`

## API Endpoints

- `POST /api/start` - Start motion detection
- `POST /api/stop` - Stop motion detection
- `GET /api/video_feed` - Stream video feed
- `GET /api/logs` - Get motion detection logs
- `GET /api/config` - Get user configuration
- `PUT /api/config` - Update user configuration
- `GET /api/health` - Health check endpoint

## Authentication

All API endpoints (except health check) require Firebase authentication. Include the Firebase ID token in the Authorization header:
