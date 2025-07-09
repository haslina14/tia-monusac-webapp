import os, uuid, logging
import threading
import traceback
import json
import time
from datetime import datetime, timedelta
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify, abort, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import subprocess

#external python scripts
patch_script_path = "patch.py"
predict_script_path = "predict.py"
merge_script_path = "merge.py"

#as db
job_status = {} 

#upload
UPLOAD_FOLDER = '/app/uploads'
#UPLOAD_FOLDER = '/mnt/c/Users/haslina.makmur/OneDrive - Cancer Research Malaysia/Documents/TIA_GUI/tia/uploads'
ALLOWED_EXTENSIONS = {'bif', 'svs', 'tif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '0e96509fcf226b1ce7a57b7a' #required for socketio
socketio = SocketIO(app, cors_allowed_origins="*")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message":"No file"})
    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message":"No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            return jsonify({
                "success": True,
                "filename": filename,
                "message": "File uploaded successfully" 
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": "File upload failed",
                "error": str(e)
            })
    return jsonify({"success": False, "message": "Invalid file type"})

@app.route('/patch', methods=['POST'])
def patch_file():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({
            "success": False, 
            "message": "No filename provided"
        })
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return jsonify({
            "success": False, 
            "message": "File not found"
        })
    
    #generate unique id
    job_id = str(uuid.uuid4())
    start_time = datetime.now()

    #initialize job status
    job_status[job_id] = {
        "status": "started",
        "filename": filename,
        "type": "patching",
        "progress": 0,
        "output": "",
        "error": "",
        "start_time": start_time.isoformat(),
        "elapsed_seconds": 0,
        "estimated_total_seconds": 600  # Default estimate: 10 minutes
    }

    #emit initial status
    socketio.emit('job_update', {
        'job_id': job_id,
        'status': job_status[job_id]
    })

    #patching in separate thread
    threading.Thread(
        target=run_patching,
        args=(job_id, file_path, filename)
    ).start()

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": "Patching started"
    })

def run_patching(job_id, file_path, filename):

    try:

        job_status[job_id]["status"] = "running"

        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })

        process = subprocess.Popen(
            ["python", patch_script_path, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        last_update_time = time.time()

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
                
            if output:
                # Process output and update status
                job_status[job_id]['output'] += output
                
                if "progress:" in output.lower():
                    try:
                        progress = float(output.split("progress:")[1].strip().rstrip("%"))
                        job_status[job_id]["progress"] = progress
                    except Exception as e:
                        logging.error(f"Error parsing progress: {e}")
                
                # Update elapsed time
                elapsed = (datetime.now() - datetime.fromisoformat(job_status[job_id]["start_time"])).total_seconds()
                job_status[job_id]["elapsed_seconds"] = int(elapsed)
                
                # Emit updates at least every 2 seconds
                if time.time() - last_update_time >= 2:
                    socketio.emit('job_update', {
                        'job_id': job_id,
                        'status': job_status[job_id]
                    })
                    last_update_time = time.time()
        
        # Final update
        job_status[job_id]["status"] = "completed" if process.returncode == 0 else "failed"
        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })
        
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })


@app.route('/predict', methods=['POST'])
def predict_file():
    data = request.get_json()
    filename = data.get('filename')

    for job_id, job in list(job_status.items()):
        if job.get('filename') == filename and job.get('type') == 'prediction':
            del job_status[job_id]

    if not filename:
        return jsonify({
            "success": False, 
            "message": "No filename provided"
        })
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return jsonify({
            "success": False, 
            "message": "File not found"
        })
    
    #generate unique job id
    job_id = str(uuid.uuid4())
    start_time = datetime.now()

    #initialize job status
    job_status[job_id] = {
        "status": "started",
        "type": "prediction",
        "filename": filename,
        "progress": 0,
        "output": "",
        "error": "",
        "start_time": start_time.isoformat(),
        "elapsed_seconds": 0,
        "estimated_total_seconds": 7200  #default estimate: longer
    }

    #emit an initail job status evnt
    socketio.emit('job_update', {
        'job_id': job_id,
        'status': job_status[job_id]
    })

    #separate thread
    threading.Thread(
        target=run_prediction,
        args=(job_id, file_path, filename)
    ).start()
    
    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": "Prediction started"
    })

#function to run prediction
def run_prediction(job_id, file_path, filename):
    try:
        job_status[job_id]["status"] = "running"
        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })

        process = subprocess.Popen(
            ["python", predict_script_path, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Create a thread to handle stderr
        def read_stderr():
            for line in process.stderr:
                job_status[job_id]["error"] += line
                socketio.emit('job_update', {
                    'job_id': job_id,
                    'status': job_status[job_id]
                })

        job_status[job_id]["status"] = "running"
        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })

        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()

        last_update_time = time.time()

        for line in process.stdout:
            line = line.strip()
            job_status[job_id]['output'] += line + '\n'
            
            if "progress:" in line.lower():
                try:
                    progress = float(line.split("progress:")[1].strip().rstrip("%"))
                    job_status[job_id]["progress"] = progress
                except Exception as e:
                    logging.error(f"Error parsing progress: {e}")
            
            # Update elapsed time
            elapsed = (datetime.now() - datetime.fromisoformat(job_status[job_id]["start_time"])).total_seconds()
            job_status[job_id]["elapsed_seconds"] = int(elapsed)
            
            # Emit updates at least every 1 second
            if time.time() - last_update_time >= 1:
                socketio.emit('job_update', {
                    'job_id': job_id,
                    'status': job_status[job_id]
                })
                last_update_time = time.time()
        
        process.wait()
        
        # Final update
        job_status[job_id]["status"] = "completed" if process.returncode == 0 else "failed"
        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })
        
    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })
        logging.error(f"Prediction failed: {str(e)}")
        logging.error(traceback.format_exc())
        
        
@app.route('/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    if job_id not in job_status:
        return jsonify({
            "success": False,
            "message": "Job not found"
        })
    
    #check if jobs has expired
    if "expire_at" in job_status[job_id]:
        try:
            expire_time = datetime.fromisoformat(job_status[job_id]["expire_at"])
            if datetime.now() > expire_time:
                #clean up
                job_data = job_status.pop(job_id)
                logging.info(f"Cleaned up expired job {job_id}")
                return jsonify({
                    "success": False,
                    "message": "Job expired and has been cleaned up",
                    "job": job_data
                })
            
        except Exception as e:
            logging.error(f"Error checking job expiration for {job_id}: {e}")
    
    if job_status[job_id]["status"] in ["started", "running"]:
        elapsed_time = (datetime.now() - datetime.fromisoformat(job_status[job_id]["start_time"])).total_seconds()
        job_status[job_id]["elapsed_seconds"] = int(elapsed_time)
    
    return jsonify({
        "success": True,
        "job": job_status[job_id]
    })

# Optional: Add a function to clean all expired jobs
def cleanup_expired_jobs():
    """Remove all expired jobs from job_status dictionary"""
    current_time = datetime.now()
    jobs_to_remove = []
    
    for j_id, job in job_status.items():
        if "expire_at" in job:
            try:
                expire_time = datetime.fromisoformat(job["expire_at"])
                if current_time > expire_time:
                    jobs_to_remove.append(j_id)
            except Exception as e:
                logging.error(f"Error checking expiration for job {j_id}: {e}")
    
    # Remove expired jobs
    for j_id in jobs_to_remove:
        del job_status[j_id]
        logging.info(f"Cleaned up expired job {j_id}")
    
    return len(jobs_to_remove)

# Optional: Add an endpoint to manually trigger cleanup
@app.route('/admin/cleanup-expired-jobs', methods=['POST'])
def admin_cleanup_expired_jobs():
    count = cleanup_expired_jobs()
    return jsonify({
        "success": True,
        "message": f"Cleaned up {count} expired jobs"
    })

@app.route('/merge', methods=['POST'])
def merge_file():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({
            "success": False, 
            "message": "No filename provided"
        })
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return jsonify({
            "success": False, 
            "message": "File not found"
        })
    
    job_id = str(uuid.uuid4())
    start_time = datetime.now()

     #initialize job status
    job_status[job_id] = {
        "status": "started",
        "type": "merging",
        "filename": filename,
        "progress": 0,
        "output": "",
        "error": "",
        "start_time": start_time.isoformat(),
        "elapsed_seconds": 0,
        "estimated_total_seconds": 3000  #default estimate: longer
    }

    #emit an initail job status evnt
    socketio.emit('job_update', {
        'job_id': job_id,
        'status': job_status[job_id]
    })

    #separate thread
    threading.Thread(
        target=run_merge,
        args=(job_id, file_path, filename)
    ).start()
    
    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": "Merge started"
    })


    
def run_merge(job_id, file_path, filename):
    try:

        job_status[job_id]["status"] = "running"

        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })

        process = subprocess.Popen(
            ["python", merge_script_path, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        output_lines = []
        error_lines = []

        def read_stderr():
            for line in process.stderr:
                error_lines.append(line.strip())
                job_status[job_id]["error"] = "\n".join(error_lines)
                socketio.emit('job_update', {
                    'job_id': job_id,
                    'status': job_status[job_id]
                })

        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()

        last_emit_time = time.time()

        for line in process.stdout:
            line = line.strip()
            output_lines.append(line)
            job_status[job_id]["output"] = "\n".join(output_lines)

            elapsed_time = (datetime.now() - datetime.fromisoformat(job_status[job_id]["start_time"])).total_seconds()
            job_status[job_id]["elapsed_seconds"] = int(elapsed_time)

            progress_updated = False

            if "progress:" in line.lower():
                try:
                    progress = float(line.split("progress:")[1].strip().rstrip("%"))
                    job_status[job_id]["progress"] = progress
                    if progress > 0:
                        job_status[job_id]["estimated_total_seconds"] = int(elapsed_time / (progress / 100.0))
                    progress_updated = True
                except Exception as e:
                    logging.error(f"Error parsing progress: {e}")

            current_time = time.time()
            if progress_updated or (current_time - last_emit_time) >= 2:
                socketio.emit('job_update', {
                    'job_id': job_id,
                    'status': job_status[job_id]
                })
                last_emit_time = current_time
        
        process.wait()

        if process.returncode == 0:
            job_status[job_id]["status"] = "completed"
            job_status[job_id]["progress"] = 100
            job_status[job_id]["expire_at"] = (datetime.now() + timedelta(hours=1)).isoformat()
        else:
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["expire_at"] = (datetime.now() + timedelta(hours=5)).isoformat()
        
        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })
    


    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        job_status[job_id]["expire_at"] = (datetime.now() + timedelta(hours=5)).isoformat()
        logging.error(f"Error in merge process: {e}")

        socketio.emit('job_update', {
            'job_id': job_id,
            'status': job_status[job_id]
        })

@app.route('/get-csv') #, methods=['GET']
def download_csv():
    filename = request.args.get('filename') 
    #base_name = filename
    base_name = os.path.splitext(filename)[0]
    try:
        result_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_Vaha")
        
        csv_name = f"nucleus_info_{base_name}_Vaha.csv"

        print(f"Looking for csv file in: {result_folder}")
        print(f"Expected file csv {csv_name}")

        csv_path = os.path.join(result_folder, csv_name)
        if not os.path.exists(csv_path):
            return jsonify({'exists': False, 'message': 'File not found'}), 404

        response = send_from_directory(
            directory=result_folder,
            path=csv_name,
            as_attachment=True,
            download_name=csv_name
        )

        #prevent cache
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response
    
    except Exception as e:
        return jsonify({'exists': False, 'message': str(e)}), 500
    

@app.route('/get-img') #, methods=['GET']
def download_img():
    filename = request.args.get('filename') 
    #base_name = filename
    base_name = os.path.splitext(filename)[0]
    try:
        result_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_Vaha")
        
        img_name = f"Merge_{base_name}_Vaha.png"

        print(f"Looking for overlay image file in: {result_dir}")
        print(f"Expected overlay image {img_name}")

        img_path = os.path.join(result_dir, img_name)
        if not os.path.exists(img_path):
            return jsonify({'exists': False, 'message': 'File not found'}), 404

        response = send_from_directory(
            directory=result_dir,
            path=img_name,
            as_attachment=True,
            download_name=img_name
        )

        #prevent cache
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response
    
    except Exception as e:
        return jsonify({'exists': False, 'message': str(e)}), 500
    
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}, Origin: {request.headers.get("Origin")}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


#if __name__ == "__main__":
 #   app.run(debug=True)
