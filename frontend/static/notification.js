function sendNotification(message) {
    if (Notification.permission === "granted") {
        new Notification(message);
    } else if (Notification.permission !== "denied") {
        Notification.requestPermission().then(permission => {
            if (permission === "granted") {
                new Notification(message);
            }
        });
    }
}

function triggerVibration() {
    if ("vibrate" in navigator) {
        navigator.vibrate([500, 200, 500]); // Vibrates for 500ms, pause, vibrates again
    }
}

function sendAlert(message) {
    sendNotification(message);
    triggerVibration();
}
