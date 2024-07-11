from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import numpy as np
import time
import nltk
from nltk.chat.util import Chat, reflections  # Add this import



# Global variable to store the detected emotion
persistent_emotion = None

def detect_emotion(face_image):
    # Placeholder function for emotion detection
    # Replace with actual emotion detection logic using your model
    emotions = ['happy', 'sad', 'neutral', 'angry', 'fear']
    return np.random.choice(emotions)

def video_stream():
    print('1')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Initialize the emotion detector outside the loop
    emotion_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()
    # Dictionary to store the count of each emotion
    emotion_counts = {}

    while time.time() - start_time < 5:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = emotion_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_face = im[y:y+h, x:x+w]

            # Detect emotion from the face region
            emotion = detect_emotion(roi_face)
            # Increment emotion count
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # Draw the detected emotion on the video feed
            cv2.putText(im, emotion, (x, y-10), font, 0.9, (0, 255, 0), 2)

        # Encode the image as JPEG
        _, jpeg = cv2.imencode('.jpg', im)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cam.release()
    # Find the emotion that occurred most frequently
    global persistent_emotion
    if emotion_counts:
        persistent_emotion = max(emotion_counts, key=emotion_counts.get)
        print('Detected Emotion (Persistent):', persistent_emotion)
    # Call redirect_based_on_emotion after the while loop
    return redirect_based_on_emotion(persistent_emotion)

def redirect_based_on_emotion(persistent_emotion):
    print("Emotion is:", persistent_emotion)
    # Check the value of the provided emotion and redirect accordingly
    if persistent_emotion == 'neutral':
        return redirect('neutral')
    elif persistent_emotion == 'sad':
        return redirect('sad')
    elif persistent_emotion == 'happy':
        return redirect('happy')
    elif persistent_emotion == 'angry':
        return redirect('angry')
    elif persistent_emotion == 'fear':
        return redirect('fear')
    else:
        return redirect('getstart')  # If no specific emotion detected or provided, redirect to a default website

def camera(request):
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def home(request):
    return render(request, 'home.html')

def contact(request):
    return render(request, 'contact.html')

def services(request):
    return render(request, 'services.html')

def blog(request):
    return render(request, 'blog.html')

def about(request):
    return render(request, 'about.html')

def getstart(request):
    return render(request, 'getstart.html')

def doctors(request):
    return render(request, 'doctors.html')

def question(request):
    return render(request, 'question.html')
    
def chatwithme(request):
    return render(request, 'chatwithme.html')    

def happy(request):
    return render(request, 'happy.html')

def sad(request):
    return render(request, 'sad.html')

def neutral(request):
    return render(request, 'neutral.html')

def rate(request):
    return render(request, 'rate.html')

def angry(request):
    return render(request, 'angry.html')

def fear(request):
    return render(request, 'fear.html')

# Define patterns and responses for the chatbot
chatbot_pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how can I help you today?",]
    ],
    [
        r"what is your name?",
        ["My name is EmoBot and I'm here to assist you with your emotions.",]
    ],
    [
        r"who are you?",
        ["I'm just a bot, but I'm here to assist you with your emotions.",]
    ],
    [
        r"(.*) (happy|excited|good|joyful)",
        ["That's great to hear!",]
    ],
    [
        r"(.*) (sad|angry|frustrated|depressed)",
        ["I'm sorry to hear that. Would you like to talk about it?",]
    ],
    [
        r"yes",
        ["Feel free to share what's on your mind.",]
    ],
    [
        r"no",
        ["Alright. Remember, I'm here whenever you need to talk.",]
    ],
    [
        r"quit",
        ["Goodbye. Take care!",]
    ],
]

# Create a chatbot instance
chatbot = Chat(chatbot_pairs, reflections)

def chat_bot_response(request):
    # Get user input from the request
    user_input = request.GET.get('user_input', '')

    # Get response from the chatbot based on user input
    response = chatbot.respond(user_input)

    # Return the response as JSON
    return JsonResponse({'response': response})