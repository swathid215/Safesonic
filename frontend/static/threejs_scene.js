// Basic Three.js scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

function playBabyCryAnimation() {
    alert("Baby is crying! 🚼");
}

function showSirenAlert() {
    alert("Siren Alert! 🚨");
}

function playGlassBreakAnimation() {
    alert("Broken Glass! 🪟");
}

camera.position.z = 5;
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();
