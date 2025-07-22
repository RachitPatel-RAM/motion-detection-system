import os
import time
import json
import base64
import datetime
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import firebase_admin
from firebase_admin import credentials, auth, db
import cloudinary
import cloudinary.uploader
import cloudinary.api
from functools import wraps
from motion_detection import MotionDetector

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Rate limiting with Redis
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.environ.get('REDIS_URL', 'redis://localhost:6379')
)

# Load environment variables or use defaults for development
def get_env_var(name, default=None):
    return os.environ.get(name, default)

# Firebase configuration
firebase_credentials = get_env_var('FIREBASE_CREDENTIALS', '{}')
firebase_db_url = get_env_var('FIREBASE_DB_URL', 'https://motion-71c2a-default-rtdb.firebaseio.com/')

# Try to parse credentials from environment variable
try:
    cred_dict = json.loads(firebase_credentials)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred, {
        'databaseURL': firebase_db_url
    })
except (json.JSONDecodeError, ValueError):
    # For development, use a temporary credential
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": "motion-71c2a",
            # Add other required fields for development
            # This is just a placeholder and won't work in production
        })
        firebase_admin.initialize_app(cred, {
            'databaseURL': firebase_db_url
        })

# Cloudinary configuration
cloudinary_config = {
    "cloud_name": get_env_var('CLOUDINARY_CLOUD_NAME', 'dpv1ulroy'),
    "api_key": get_env_var('CLOUDINARY_API_KEY', '753843896383315'),
    "api_secret": get_env_var('CLOUDINARY_API_SECRET', 'MSmNF__TeFRS97eghntdWZksArE'),
    "upload_preset": get_env_var('CLOUDINARY_UPLOAD_PRESET', 'motion')
}

cloudinary.config(
    cloud_name=cloudinary_config["cloud_name"],
    api_key=cloudinary_config["api_key"],
    api_secret=cloudinary_config["api_secret"]
)

# Email configuration
email_config = {
    "sender_email": get_env_var('EMAIL_SENDER', 'motiondectedobject@gmail.com'),
    "app_password": get_env_var('EMAIL_APP_PASSWORD', 'fybizhsv hthk vvbh')
}

# JWT Secret
jwt_secret = get_env_var('JWT_SECRET', 'r1dXIDemSne82FHr3mAbm')

# Global motion detector instance
motion_detector = None
motion_thread = None
motion_running = False

# User configurations cache
user_configs = {}

# Token verification decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
            
        try:
            # Verify the Firebase token
            decoded_token = auth.verify_id_token(token)
            user_id = decoded_token['uid']
            
            # Add user_id to kwargs
            kwargs['user_id'] = user_id
            
            # Load user config if not cached
            if user_id not in user_configs:
                load_user_config(user_id)
                
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'message': f'Token is invalid: {str(e)}'}), 401
            
    return decorated

# Load user configuration from Firebase
def load_user_config(user_id):
    try:
        user_ref = db.reference(f'/users/{user_id}')
        user_data = user_ref.get()
        
        if not user_data:
            # Initialize user data if not exists
            user_data = {
                "email": "",  # Will be updated on first login
                "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
                "sensitivity": {"slow": 50, "fast": 70},
                "storage_preference": "local",
                "daily_cloud_count": 0,
                "last_reset_date": datetime.datetime.now().strftime('%Y-%m-%d')
            }
            user_ref.set(user_data)
            
        # Convert ROI to tuple format for motion detector
        roi = None
        if user_data.get('roi'):
            roi_data = user_data['roi']
            if all(k in roi_data for k in ['x', 'y', 'width', 'height']):
                roi = (roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height'])
                
        # Convert sensitivity settings
        sensitivity = user_data.get('sensitivity', {})
        slow_sensitivity = sensitivity.get('slow', 50)
        fast_sensitivity = sensitivity.get('fast', 70)
        
        # Calculate actual threshold values based on sensitivity
        # Lower sensitivity number = higher threshold (less sensitive)
        threshold_sensitivity = max(5, 30 - (slow_sensitivity / 5))  # Range: 5-25
        min_contour_area = max(50, 200 - (slow_sensitivity * 1.5))  # Range: 50-150
        speed_threshold = max(50, 150 - (fast_sensitivity))  # Range: 50-150
        
        # Store in cache
        user_configs[user_id] = {
            "email": user_data.get('email', ''),
            "roi": roi,
            "threshold_sensitivity": int(threshold_sensitivity),
            "min_contour_area": int(min_contour_area),
            "speed_threshold": int(speed_threshold),
            "storage_preference": user_data.get('storage_preference', 'local'),
            "daily_cloud_count": user_data.get('daily_cloud_count', 0),
            "last_reset_date": user_data.get('last_reset_date', '')
        }
        
        # Reset daily count if needed
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if user_configs[user_id]["last_reset_date"] != today:
            user_configs[user_id]["daily_cloud_count"] = 0
            user_configs[user_id]["last_reset_date"] = today
            user_ref.update({
                "daily_cloud_count": 0,
                "last_reset_date": today
            })
            
        return user_configs[user_id]
    except Exception as e:
        print(f"Error loading user config: {str(e)}")
        return None

# Motion detection callback
def on_motion_detected(motion_data, user_id):
    try:
        # Get user config
        user_config = user_configs.get(user_id)
        if not user_config:
            print(f"No config found for user {user_id}")
            return
            
        # Prepare data for logging
        timestamp = motion_data["timestamp"]
        snapshot_path = motion_data["snapshot_path"]
        video_path = motion_data["video_path"]
        is_speedy = motion_data["is_speedy"]
        object_size = motion_data["object_size"]
        
        # Determine storage preference
        storage_preference = user_config["storage_preference"]
        snapshot_urls = []
        
        # Handle cloud storage if selected
        if storage_preference == "cloud":
            # Check daily limit
            if user_config["daily_cloud_count"] >= 30:
                print(f"Daily cloud upload limit reached for user {user_id}")
                storage_preference = "local"  # Fallback to local
            else:
                # Upload to Cloudinary
                try:
                    upload_result = cloudinary.uploader.upload(
                        snapshot_path,
                        folder=f"motion_detection/{user_id}",
                        public_id=f"snapshot_{int(time.time())}"
                    )
                    snapshot_urls.append(upload_result["secure_url"])
                    
                    # Update daily count
                    user_config["daily_cloud_count"] += 1
                    db.reference(f'/users/{user_id}').update({
                        "daily_cloud_count": user_config["daily_cloud_count"]
                    })
                except Exception as e:
                    print(f"Cloudinary upload error: {str(e)}")
                    storage_preference = "local"  # Fallback to local
        
        # If local or cloud failed, use local path
        if storage_preference == "local":
            snapshot_urls = [snapshot_path]  # Use local path
            
        # Log to Firebase
        log_entry = {
            "timestamp": timestamp,
            "snapshots": snapshot_urls,
            "motion_type": "Speedy" if is_speedy else "Normal",
            "object_size": object_size,
            "storage": storage_preference
        }
        
        # Add to user's logs
        log_ref = db.reference(f'/users/{user_id}/logs').push()
        log_ref.set(log_entry)
        
        # Send email alert if we have the user's email
        user_email = user_config.get("email")
        if user_email:
            send_email_alert(user_email, log_entry, snapshot_path)
            
    except Exception as e:
        print(f"Error in motion callback: {str(e)}")

# Send email alert with snapshots
def send_email_alert(recipient_email, log_entry, snapshot_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = email_config["sender_email"]
        msg['To'] = recipient_email
        msg['Subject'] = f"Motion Alert - {log_entry['timestamp']}"
        
        # Create HTML content
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4285f4; color: white; padding: 10px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ font-size: 12px; color: #777; text-align: center; padding: 10px; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                td, th {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Motion Detection Alert</h2>
                </div>
                <div class="content">
                    <p>Motion has been detected by your security system.</p>
                    
                    <table>
                        <tr>
                            <th>Date & Time:</th>
                            <td>{log_entry['timestamp']}</td>
                        </tr>
                        <tr>
                            <th>Motion Type:</th>
                            <td>{log_entry['motion_type']}</td>
                        </tr>
                        <tr>
                            <th>Object Size:</th>
                            <td>{log_entry['object_size']}</td>
                        </tr>
                        <tr>
                            <th>Storage:</th>
                            <td>{log_entry['storage']}</td>
                        </tr>
                    </table>
                    
                    <h3>Snapshot:</h3>
                    <img src="cid:motion_snapshot" alt="Motion Detection Snapshot">
                </div>
                <div class="footer">
                    <p>This is an automated message from your Motion Detection System.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        # Attach image
        with open(snapshot_path, 'rb') as f:
            img_data = f.read()
            image = MIMEImage(img_data)
            image.add_header('Content-ID', '<motion_snapshot>')
            msg.attach(image)
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(email_config["sender_email"], email_config["app_password"])
            server.send_message(msg)
            
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Motion detection thread function
def motion_detection_thread(user_id):
    global motion_detector, motion_running
    
    try:
        # Get user configuration
        user_config = user_configs.get(user_id)
        if not user_config:
            print(f"No config found for user {user_id}")
            motion_running = False
            return
            
        # Configure motion detector
        config = {
            "threshold_sensitivity": user_config["threshold_sensitivity"],
            "min_contour_area": user_config["min_contour_area"],
            "speed_threshold": user_config["speed_threshold"],
            "roi": user_config["roi"]
        }
        
        motion_detector = MotionDetector(config)
        
        # Set callback
        motion_detector.on_motion_detected = lambda data: on_motion_detected(data, user_id)
        
        # Start capture
        motion_detector.start_capture()
        
        # Process frames until stopped
        while motion_running:
            frame, motion_detected, motion_data = motion_detector.process_frame()
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        print(f"Error in motion thread: {str(e)}")
    finally:
        if motion_detector:
            motion_detector.stop()
        motion_running = False

# API Routes
@app.route('/api/start', methods=['POST'])
@token_required
def start_detection(user_id):
    global motion_thread, motion_running
    
    if motion_running:
        return jsonify({'message': 'Motion detection already running'}), 400
        
    # Start motion detection in a separate thread
    motion_running = True
    motion_thread = threading.Thread(target=motion_detection_thread, args=(user_id,))
    motion_thread.daemon = True
    motion_thread.start()
    
    return jsonify({'message': 'Motion detection started'})

@app.route('/api/stop', methods=['POST'])
@token_required
def stop_detection(user_id):
    global motion_running
    
    if not motion_running:
        return jsonify({'message': 'Motion detection not running'}), 400
        
    # Stop the motion detection thread
    motion_running = False
    if motion_thread:
        motion_thread.join(timeout=5.0)
        
    return jsonify({'message': 'Motion detection stopped'})

@app.route('/api/video_feed')
@token_required
def video_feed(user_id):
    def generate():
        while motion_running and motion_detector:
            frame = motion_detector.get_frame_jpeg()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
@token_required
def get_logs(user_id):
    try:
        # Get logs from Firebase
        logs_ref = db.reference(f'/users/{user_id}/logs')
        logs = logs_ref.get()
        
        if not logs:
            return jsonify([])
            
        # Convert to list and sort by timestamp (newest first)
        logs_list = []
        for log_id, log_data in logs.items():
            log_data['id'] = log_id
            logs_list.append(log_data)
            
        logs_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify(logs_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'PUT'])
@token_required
def user_config(user_id):
    if request.method == 'GET':
        # Return current config
        user_config = load_user_config(user_id)  # Reload to ensure latest
        if not user_config:
            return jsonify({'error': 'Failed to load user configuration'}), 500
            
        return jsonify({
            'roi': user_config['roi'],
            'sensitivity': {
                'slow': 100 - int(user_config['threshold_sensitivity'] * 5),  # Convert back to 1-100 scale
                'fast': 100 - int((user_config['speed_threshold'] - 50) / 1)  # Convert back to 1-100 scale
            },
            'storage_preference': user_config['storage_preference'],
            'daily_cloud_count': user_config['daily_cloud_count']
        })
    else:  # PUT
        try:
            data = request.json
            updates = {}
            
            # Update ROI if provided
            if 'roi' in data:
                roi = data['roi']
                if all(k in roi for k in ['x', 'y', 'width', 'height']):
                    updates['roi'] = roi
                    
            # Update sensitivity if provided
            if 'sensitivity' in data:
                sensitivity = data['sensitivity']
                updates['sensitivity'] = sensitivity
                
            # Update storage preference if provided
            if 'storage_preference' in data:
                pref = data['storage_preference']
                if pref in ['local', 'cloud']:
                    updates['storage_preference'] = pref
                    
            # Update in Firebase
            if updates:
                db.reference(f'/users/{user_id}').update(updates)
                # Reload config
                load_user_config(user_id)
                
            return jsonify({'message': 'Configuration updated'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/health')
@limiter.limit("10 per minute")
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.datetime.now().isoformat()})

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False, threaded=True)
