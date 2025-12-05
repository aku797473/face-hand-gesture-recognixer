
import {
  FilesetResolver,
  GestureRecognizer,
  FaceLandmarker,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js";


const video = document.getElementById("video");
const landmarkCanvas = document.getElementById("landmark");
const landmarkCtx = landmarkCanvas.getContext("2d", { willReadFrequently: true });
const drawingCanvas = document.getElementById("drawing");
const drawingCtx = drawingCanvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const pinchToggle = document.getElementById("pinchToggle");
const smileToggle = document.getElementById("smileToggle");
const colorPicker = document.getElementById("colorPicker");
const brushSize = document.getElementById("brushSize");

const feedCanvas = document.createElement("canvas");
const feedCtx = feedCanvas.getContext("2d");


let gestureVideo = null;   // runningMode: "video"
let gestureImage = null;   // runningMode: "image" fallback
let faceVideo = null;      // runningMode: "video"
let faceImage = null;      // runningMode: "image" fallback

let drawingUtils = null;

let running = false;
let prevX = null, prevY = null;
const SMOOTHING = 0.45;
let PINCH_THRESHOLD = 30; // pixels — may adjust per resolution


function setBrush() {
  drawingCtx.lineWidth = parseFloat(brushSize.value) || 4;
  drawingCtx.strokeStyle = colorPicker.value || "#ff0000";
  drawingCtx.lineCap = "round";
  drawingCtx.lineJoin = "round";
}
setBrush();
colorPicker.addEventListener("input", setBrush);
brushSize.addEventListener("input", setBrush);

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 }, audio: false
  });
  video.srcObject = stream;
  await video.play();
  resizeCanvases();
}


function resizeCanvases() {
  const w = video.videoWidth || video.clientWidth || 640;
  const h = video.videoHeight || video.clientHeight || 480;
  [landmarkCanvas, drawingCanvas, feedCanvas].forEach(c => {
    c.width = w; c.height = h;
  });
  // tune pinch threshold relative to size (approx)
  PINCH_THRESHOLD = Math.max(18, Math.round(Math.min(w,h) * 0.03));
}


async function loadModels() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  // Gesture recognizer — video
  gestureVideo = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task",
    },
    runningMode: "video",
    numHands: 2,
  });

  // Gesture recognizer — image (fallback)
  gestureImage = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task",
    },
    runningMode: "image",
    numHands: 2,
  });

  // Face landmarker — video
  faceVideo = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    },
    runningMode: "video",
    numFaces: 2,
  });

  // Face landmarker — image (fallback)
  faceImage = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    },
    runningMode: "image",
    numFaces: 2,
  });

  drawingUtils = new DrawingUtils(landmarkCtx);
}

// Drawing helpers
function smoothDraw(x, y) {
  if (prevX === null || prevY === null) {
    prevX = x; prevY = y; return;
  }
  const nx = prevX + SMOOTHING * (x - prevX);
  const ny = prevY + SMOOTHING * (y - prevY);

  drawingCtx.beginPath();
  drawingCtx.moveTo(prevX, prevY);
  drawingCtx.lineTo(nx, ny);
  drawingCtx.stroke();

  prevX = nx; prevY = ny;
}

function resetPrev() { prevX = prevY = null; }
function clearDrawing() { drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height); }


function isSmiling(landmarks) {
  const left = landmarks[61];
  const right = landmarks[291];
  if (!left || !right) return false;
  return Math.abs(right.x - left.x) > 0.06; // tuned threshold (normalized coords)
}


function mouthOpen(landmarks) {
  const top = landmarks[13];
  const bottom = landmarks[14];
  if (!top || !bottom) return false;
  return Math.abs(bottom.y - top.y) > 0.03;
}


async function processLoop() {
  if (!running) return;
  resizeCanvases();
  landmarkCtx.clearRect(0,0,landmarkCanvas.width,landmarkCanvas.height);

  const ts = Math.round(performance.now());


  const hasVideoFrame = typeof VideoFrame === "function";

  
  try {
    let handResult = null;
    if (hasVideoFrame && gestureVideo) {
      // fast path using video-mode recognizer
      handResult = gestureVideo.recognizeForVideo(video, ts);
    } else if (gestureImage) {
      // fallback: send an image (offscreen canvas)
      feedCtx.drawImage(video, 0, 0, feedCanvas.width, feedCanvas.height);
      handResult = gestureImage.recognize(feedCanvas);
    }

    if (handResult?.landmarks) {
      for (let i=0;i<handResult.landmarks.length;i++){
        const hand = handResult.landmarks[i];
        if (!hand) continue;
        drawingUtils.drawConnectors(hand, GestureRecognizer.HAND_CONNECTIONS, { color: "cyan", lineWidth: 2 });
        drawingUtils.drawLandmarks(hand, { color: "white", radius: 2 });
      }

      // built-in gesture name (if present)
      const gestureName = handResult?.gestures?.[0]?.[0]?.categoryName;
      if (gestureName === "Thumb_Down") { clearDrawing(); resetPrev(); }

      // pinch detection on first hand (thumb 4, index 8)
      if (pinchToggle.checked && handResult.landmarks[0]) {
        const hand0 = handResult.landmarks[0];
        const thumb = hand0[4], index = hand0[8];
        if (thumb && index) {
          const dx = (thumb.x - index.x) * drawingCanvas.width;
          const dy = (thumb.y - index.y) * drawingCanvas.height;
          const dist = Math.sqrt(dx*dx + dy*dy);
          if (dist <= PINCH_THRESHOLD) {
            const x = index.x * drawingCanvas.width;
            const y = index.y * drawingCanvas.height;
            setBrush(); smoothDraw(x, y);
          } else resetPrev();
        }
      }
    }
  } catch (err) {
    
  }

  
  try {
    let faceRes = null;
    if (hasVideoFrame && faceVideo) {
      try {
        faceRes = faceVideo.detectForVideo(video, ts);
      } catch (e) {
        // fallback: create explicit VideoFrame if supported
        try {
          const vf = new VideoFrame(video);
          faceRes = faceVideo.detectForVideo(vf, ts);
          vf.close();
        } catch (err) {
          
          faceRes = null;
        }
      }
    }

    if (!faceRes && faceImage) {
      feedCtx.drawImage(video, 0, 0, feedCanvas.width, feedCanvas.height);
      faceRes = faceImage.detect(feedCanvas);
    }

    if (faceRes?.faceLandmarks?.[0]) {
      const fl = faceRes.faceLandmarks[0];
      try {
        for (const p of fl) {
          landmarkCtx.beginPath();
          landmarkCtx.arc(p.x * landmarkCanvas.width, p.y * landmarkCanvas.height, 2, 0, Math.PI*2);
          landmarkCtx.fillStyle = "rgba(0,200,255,0.8)";
          landmarkCtx.fill();
        }
      } catch (e) {  }

      
      if (smileToggle.checked && isSmiling(fl)) {
        const nose = fl[1] || fl[4];
        if (nose) {
          setBrush();
          smoothDraw(nose.x * drawingCanvas.width, nose.y * drawingCanvas.height);
        }
      }

      
      const mo = mouthOpen(fl);
      drawingCtx.lineWidth = mo ? Math.max(6, parseFloat(brushSize.value)) : parseFloat(brushSize.value);
    }
  } catch (err) {
    // console.warn("face frame error", err);
  }

  requestAnimationFrame(processLoop);
}


startBtn.addEventListener("click", async () => {
  try {
    startBtn.disabled = true;
    await startCamera();
    await loadModels();
    running = true;
    stopBtn.disabled = false;
    processLoop();
  } catch (err) {
    console.error("Start failed:", err);
    alert("Start failed — open console. " + (err && err.message));
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener("click", () => {
  running = false;
  stopBtn.disabled = true;
  startBtn.disabled = false;
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
});

clearBtn.addEventListener("click", () => { clearDrawing(); resetPrev(); });