# FitMatch Setup Guide
Here’s a professional and informative `README.md` for your **FitMatch** project:

---

# 🏋️‍♂️ FitMatch

FitMatch is an AI-powered web application that enables users to compete in real-time fitness challenges like **squats**, **pushups**, **planks**, and **high knees** using their webcam. It leverages computer vision (MediaPipe & OpenCV) to analyze user movements and evaluate exercise performance in real-time.

---

## 🚀 Features

* 🔍 Real-time pose detection using **MediaPipe**
* 📸 Webcam-based fitness tracking for:

  * Squats
  * Pushups
  * Planks
  * High Knees
* 🧠 Smart exercise analysis logic based on joint angles
* 📈 Counter and timer logic to evaluate and score performance
* 🧩 Modular design for easy extension with new exercises

---

## 🧰 Tech Stack

* **Backend**: Flask
* **Computer Vision**: OpenCV, MediaPipe
* **Frontend**: HTML/CSS (via `render_template`)
* **Utilities**: NumPy, PIL, base64
* **Pose Analysis**: Custom thresholds defined per exercise

---

## 📂 Project Structure

```
fitmatch/
├── app.py                   # Main Flask app
├── templates/
│   └── index.html           # Frontend HTML
├── thresholds/
│   ├── squat.py             # Squat analysis logic
│   ├── plank.py             # Plank timer logic
│   ├── pushup.py            # Pushup analysis logic
│   └── high_knees.py        # High Knees analysis logic
```

---

## 🧪 How It Works

1. User accesses the frontend (`index.html`) and selects an exercise.
2. The webcam captures frames and sends them to the Flask server via `/process_frame`.
3. Flask decodes the image and analyzes it using MediaPipe and OpenCV.
4. The corresponding exercise analysis logic is applied to count reps or time.
5. A processed image is returned and displayed on the client.

---

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fitmatch.git
cd fitmatch
```

### 2. Install Dependencies

Make sure you have Python 3.7+ and then run:

```bash
pip install -r requirements.txt
```

> Sample `requirements.txt`:

```text
flask
opencv-python
mediapipe
numpy
pillow
```

### 3. Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser.

---

## 🎯 Future Enhancements

* Multiplayer fitness challenges using WebRTC or socket communication
* User authentication and leaderboards
* Gamified rewards and virtual store integration
* ML-based feedback on exercise posture

---

## 🤝 Contributing

We welcome contributions! Feel free to fork the repository and submit pull requests.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for more details.

---

Would you like me to generate a logo or diagram for FitMatch as well?

## Setup Virtual Environment

``` 
python -m venv fitmatch_env 
```
``` 
fitmatch_env\Scripts\activate 
```
``` 
pip install -r requirements.txt 
```
``` 
cd backend 
```
``` 
python main.py 
```

then in browser goto localhost:5000

in thresholds folder add exercises with template similar to squat_analysis 
