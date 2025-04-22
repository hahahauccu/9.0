const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const restartBtn = document.getElementById('restartBtn');
const poseImage = document.getElementById('poseImage');

let detector, rafId;
let currentPoseIndex = 0;
const totalPoses = 7;
let standardKeypointsList = [];
let poseOrder = [];

// 隨機打亂順序
function shufflePoseOrder() {
  poseOrder = Array.from({ length: totalPoses }, (_, i) => i + 1);
  for (let i = poseOrder.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [poseOrder[i], poseOrder[j]] = [poseOrder[j], poseOrder[i]];
  }
}

// 載入 PNG or png
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

// 載入所有姿勢
async function loadStandardKeypoints() {
  standardKeypointsList = [];
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

// 計算角度
function computeAngle(a, b, c) {
  const ab = { x: b.x - a.x, y: b.y - a.y };
  const cb = { x: b.x - c.x, y: b.y - c.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const abLen = Math.hypot(ab.x, ab.y);
  const cbLen = Math.hypot(cb.x, cb.y);
  const angleRad = Math.acos(dot / (abLen * cbLen));
  return angleRad * (180 / Math.PI);
}

// 角度比對
function compareKeypointsAngleBased(user, standard) {
  const angles = [
    ["left_shoulder", "left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow", "right_wrist"],
    ["left_hip", "left_knee", "left_ankle"],
    ["right_hip", "right_knee", "right_ankle"],
    ["left_elbow", "left_shoulder", "left_hip"],
    ["right_elbow", "right_shoulder", "right_hip"]
  ];

  let totalDiff = 0, count = 0;

  for (const [aName, bName, cName] of angles) {
    const aUser = user.find(kp => kp.name === aName);
    const bUser = user.find(kp => kp.name === bName);
    const cUser = user.find(kp => kp.name === cName);
    const aStd = standard.find(kp => kp.name === aName);
    const bStd = standard.find(kp => kp.name === bName);
    const cStd = standard.find(kp => kp.name === cName);

    if ([aUser, bUser, cUser, aStd, bStd, cStd].every(kp => kp?.score > 0.5)) {
      const angleUser = computeAngle(aUser, bUser, cUser);
      const angleStd = computeAngle(aStd, bStd, cStd);
      totalDiff += Math.abs(angleUser - angleStd);
      count++;
    }
  }

  if (count === 0) return 0;
  const avgDiff = totalDiff / count;
  return avgDiff < 7.5 ? 1 : 0;
}

// 畫骨架
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

// 偵測主邏輯
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
    if (sim === 1) {
      currentPoseIndex++;
      if (currentPoseIndex < totalPoses) {
        poseImage.src = standardKeypointsList[currentPoseIndex].imagePath;
      } else {
        cancelAnimationFrame(rafId);
        poseImage.src = "";
        alert('🎉 全部完成！');
        restartBtn.style.display = 'block';
      }
    }
  }

  rafId = requestAnimationFrame(detect);
}

// 啟動遊戲
async function startGame() {
  cancelAnimationFrame(rafId);
  standardKeypointsList = [];
  currentPoseIndex = 0;
  startBtn.style.display = 'none';
  restartBtn.style.display = 'none';
  poseImage.src = "";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { exact: 'environment' }, // ✅ 主鏡頭
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    alert("⚠️ 無法開啟攝影機：" + err.message);
    console.error(err);
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0); // ✅ 鏡像

  try {
    await tf.setBackend('webgl');
    await tf.ready();
  } catch {
    await tf.setBackend('wasm');
    await tf.ready();
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
restartBtn.addEventListener("click", startGame);

// 點畫面也能跳過關卡
document.body.addEventListener('click', () => {
  if (!standardKeypointsList.length) return;
  currentPoseIndex++;
  if (currentPoseIndex < totalPoses) {
    poseImage.src = standardKeypointsList[currentPoseIndex].imagePath;
  } else {
    cancelAnimationFrame(rafId);
    alert('🎉 全部完成！');
    restartBtn.style.display = 'block';
  }
});
