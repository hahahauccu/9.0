// ✅ Pose Matching Game with Angle-Based Comparison
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const poseImage = document.getElementById('poseImage');

let detector, rafId;
let currentPoseIndex = 0;
const totalPoses = 7;
const similarityThreshold = 0.85; // 角度比對的門檻（可調整）
let standardKeypointsList = [];
let poseOrder = [];

function shufflePoseOrder() {
  poseOrder = Array.from({ length: totalPoses }, (_, i) => i + 1);
  for (let i = poseOrder.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [poseOrder[i], poseOrder[j]] = [poseOrder[j], poseOrder[i]];
  }
  console.log("本次順序：", poseOrder);
}

function resolvePoseImageName(base) {
  const png = `poses/${base}.png`;
  const PNG = `poses/${base}.PNG`;
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(png);
    img.onerror = () => resolve(PNG);
    img.src = png;
  });
}

async function loadStandardKeypoints() {
  for (const i of poseOrder) {
    const res = await fetch(`poses/pose${i}.json`);
    const json = await res.json();
    const keypoints = json.keypoints || json;
    standardKeypointsList.push({
      id: i,
      keypoints,
      imagePath: await resolvePoseImageName(`pose${i}`)
    });
  }
}

function drawKeypoints(kps, color, radius, alpha) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  kps.forEach(kp => {
    if (kp.score > 0.4) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, radius, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
  ctx.globalAlpha = 1.0;
}

function computeAngle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);
  if (magAB * magCB === 0) return 0;
  let cosine = dot / (magAB * magCB);
  cosine = Math.max(-1, Math.min(1, cosine));
  return Math.acos(cosine) * (180 / Math.PI);
}

function compareKeypointsAngleBased(userKeypoints, standardKeypoints) {
  const get = name => userKeypoints.find(k => k.name === name);
  const getStd = name => standardKeypoints.find(k => k.name === name);

  const anglesToCompare = [
    ["left_shoulder", "left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow", "right_wrist"],
    ["left_hip", "left_knee", "left_ankle"],
    ["right_hip", "right_knee", "right_ankle"],
    ["left_elbow", "left_shoulder", "left_hip"],
    ["right_elbow", "right_shoulder", "right_hip"]
  ];

  let totalDiff = 0;
  let count = 0;

  for (const [a, b, c] of anglesToCompare) {
    const ua = get(a), ub = get(b), uc = get(c);
    const sa = getStd(a), sb = getStd(b), sc = getStd(c);

    if ([ua, ub, uc, sa, sb, sc].every(kp => kp && kp.score > 0.4)) {
      const userAngle = computeAngle(ua, ub, uc);
      const stdAngle = computeAngle(sa, sb, sc);
      const diff = Math.abs(userAngle - stdAngle);
      totalDiff += diff;
      count++;
    }
  }

  if (count === 0) return 0;
  const avgDiff = totalDiff / count;
  const similarity = 1 - (avgDiff / 60);
  return Math.max(0, Math.min(1, similarity));
}

async function detect() {
  const result = await detector.estimatePoses(video);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const currentPose = standardKeypointsList[currentPoseIndex];
  if (currentPose) drawKeypoints(currentPose.keypoints, 'blue', 6, 0.5);

  if (result.length > 0) {
    const user = result[0].keypoints;
    drawKeypoints(user, 'red', 6, 1.0);

    const sim = compareKeypointsAngleBased(user, currentPose.keypoints);
    if (sim > similarityThreshold) {
      currentPoseIndex++;
      if (currentPoseIndex < totalPoses) {
        poseImage.src = standardKeypointsList[currentPoseIndex].imagePath;
      } else {
        cancelAnimationFrame(rafId);
        alert('🎉 全部完成！');
        return;
      }
    }
  }

  rafId = requestAnimationFrame(detect);
}

async function startGame() {
  startBtn.disabled = true;
  startBtn.style.display = 'none';

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { exact: 'environment' },
      width: { ideal: 640 },
      height: { ideal: 480 }
    },
    audio: false
  });
  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);

  try {
    await tf.setBackend('webgl'); await tf.ready();
  } catch {
    try {
      await tf.setBackend('wasm'); await tf.ready();
    } catch {
      await tf.setBackend('cpu'); await tf.ready();
    }
  }

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );

  shufflePoseOrder();
  await loadStandardKeypoints();
  poseImage.src = standardKeypointsList[0].imagePath;
  detect();
}

startBtn.addEventListener("click", startGame);

document.body.addEventListener('click', () => {
  if (!standardKeypointsList.length) return;
  currentPoseIndex++;
  if (currentPoseIndex < totalPoses) {
    poseImage.src = standardKeypointsList[currentPoseIndex].imagePath;
  } else {
    cancelAnimationFrame(rafId);
    alert('🎉 全部完成！');
  }
});
