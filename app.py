from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from keras.models import load_model
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import threading

app = Flask(__name__, static_url_path='/static')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def send_email(subject, body, frames_folder):
    sender_email = 'durgaprasaddiviti@gmail.com'  # Sender's email address
    receiver_emails = ['sandeshramishetty3620@gmail.com']  # List of recipients
    password = 'kivo ckgp cjys xreg'  # Sender's email password

    msg = MIMEMultipart('mixed') #difeerent content types 
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = ', '.join(receiver_emails)

    text = body
    html = """\
    <html>
      <body>
        <p>{}</p>
      </body>
    </html>
    """.format(body)

    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')

    msg.attach(part1)
    msg.attach(part2)

    for frame_file in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_file)
        if os.path.isfile(frame_path):
            with open(frame_path, 'rb') as attachment:
                image = MIMEImage(attachment.read())
                image.add_header('Content-Disposition', 'attachment', filename=frame_file)
                msg.attach(image)

    # Send the message via SMTP server.
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_emails, msg.as_string())
    server.quit()

def send_email_async(subject, body, frames_folder):
    email_thread = threading.Thread(target=send_email, args=(subject, body, frames_folder))
    email_thread.start()

def predict_violence(video_path, model):
    # Function to predict violence in a video and send email with frames
    IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
    SEQUENCE_LENGTH = 16
    NUM_FRAMES_TO_SEND = 3  

    def frames_extraction(video_path):
        frames_list = []
        video_reader = cv2.VideoCapture(video_path)
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
        for frame_counter in range(SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)
        video_reader.release()
        return frames_list, skip_frames_window

    print("Extracting frames from video...")
    frames, skip_frames_window = frames_extraction(video_path)
    print("Number of frames extracted:", len(frames))

    if len(frames) == SEQUENCE_LENGTH:
        frames = np.asarray(frames)
        frames = np.expand_dims(frames, axis=0)
        print("Performing prediction...")
        prediction = model.predict(frames)
        print("Prediction probabilities:", prediction)
        predicted_label = np.argmax(prediction)
        print("Predicted label index:", predicted_label)
        predicted_class = "Violence" if predicted_label == 1 else "Non-violence"
        print("Predicted class:", predicted_class)
        if predicted_class == "Violence":
            output_folder = "violence_frames"
            for file in os.listdir(output_folder):
                file_path = os.path.join(output_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            video_reader = cv2.VideoCapture(video_path)
            frame_counter = 0
            while True:
                success, frame = video_reader.read()
                if not success:
                    break
                if frame_counter % skip_frames_window == 0:
                    frame_path = os.path.join(output_folder, f"frame_{frame_counter}.jpg")
                    cv2.imwrite(frame_path, frame)
                frame_counter += 1
            video_reader.release()
            # Send email notification with attached frames
            send_email_async("Crime Detected", "Crime has been detected in the uploaded video. Please find attached frames.",
                       frames_folder=output_folder)
        return predicted_class
    else:
        print("Invalid video length")
        return "Invalid video length"


model = load_model('violence.h5',compile=False)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result = predict_violence(file_path, load_model('violence.h5'))
        return render_template('result.html', result=result)
    else:
        return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)