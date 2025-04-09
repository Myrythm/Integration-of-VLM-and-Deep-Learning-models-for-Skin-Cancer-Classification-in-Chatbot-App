function createSnowflake() {
    const snowflake = document.createElement('div');
    snowflake.classList.add('snowflake');
    snowflake.style.left = Math.random() * 100 + '%';
    snowflake.style.animation = `fall ${Math.random() * 3 + 5}s linear`;
    
    const container = document.querySelector('.snowfall-container');
    container.appendChild(snowflake);
    
    // Remove snowflake after animation
    snowflake.addEventListener('animationend', () => {
        snowflake.remove();
    });
}

// Create snowflakes at intervals
function startSnowfall() {
    setInterval(createSnowflake, 200); // Adjust interval for more/less density
}

// Start when page loads
document.addEventListener('DOMContentLoaded', startSnowfall);