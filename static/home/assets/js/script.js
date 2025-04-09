(function() {
  "use strict";

  /**
   * Main initialization function
   */
  function initializeWebsite() {
      try {
          initScrollHandler();
          initMobileNav();
          initDropdowns();
          initScrollTop();
          initAOS();
          initStatCounters();
          initScrollSpy();
          initParticleEffect();
          initSmoothScroll();
          createFloatingElements();
            animateCounters();
      } catch (error) {
          console.error('Error during website initialization:', error);
      }
  }

  /**
   * Scroll handler for header
   */
  function initScrollHandler() {
      const selectBody = document.querySelector('body');
      const selectHeader = document.querySelector('#header');
      
      if (!selectHeader) {
          console.warn('Header element not found');
          return;
      }

      function toggleScrolled() {
          if (!selectHeader.classList.contains('scroll-up-sticky') && 
              !selectHeader.classList.contains('sticky-top') && 
              !selectHeader.classList.contains('fixed-top')) return;
              
          window.scrollY > 100 ? 
              selectBody.classList.add('scrolled') : 
              selectBody.classList.remove('scrolled');
      }

      document.addEventListener('scroll', toggleScrolled);
      window.addEventListener('load', toggleScrolled);

      // Cleanup function
      return () => {
          document.removeEventListener('scroll', toggleScrolled);
          window.removeEventListener('load', toggleScrolled);
      };
  }

  /**
   * Mobile navigation handler
   */
  function initMobileNav() {
      const mobileNavToggleBtn = document.querySelector('.mobile-nav-toggle');
      if (!mobileNavToggleBtn) {
          console.warn('Mobile nav toggle button not found');
          return;
      }

      function mobileNavToogle() {
          const body = document.querySelector('body');
          body.classList.toggle('mobile-nav-active');
          mobileNavToggleBtn.classList.toggle('bi-list');
          mobileNavToggleBtn.classList.toggle('bi-x');
      }

      mobileNavToggleBtn.addEventListener('click', mobileNavToogle);

      // Handle navigation links
      const navMenuLinks = document.querySelectorAll('#navmenu a');
      navMenuLinks.forEach(link => {
          link.addEventListener('click', () => {
              if (document.querySelector('.mobile-nav-active')) {
                  mobileNavToogle();
              }
          });
      });

      // Cleanup function
      return () => {
          mobileNavToggleBtn.removeEventListener('click', mobileNavToogle);
          navMenuLinks.forEach(link => {
              link.removeEventListener('click', mobileNavToogle);
          });
      };
  }

  /**
   * Dropdown menu handler
   */
  function initDropdowns() {
      const dropdownToggles = document.querySelectorAll('.navmenu .toggle-dropdown');
      
      function handleDropdownToggle(e) {
          e.preventDefault();
          this.parentNode.classList.toggle('active');
          const dropdown = this.parentNode.nextElementSibling;
          if (dropdown) {
              dropdown.classList.toggle('dropdown-active');
          }
          e.stopImmediatePropagation();
      }

      dropdownToggles.forEach(toggle => {
          toggle.addEventListener('click', handleDropdownToggle);
      });

      // Cleanup function
      return () => {
          dropdownToggles.forEach(toggle => {
              toggle.removeEventListener('click', handleDropdownToggle);
          });
      };
  }

  /**
   * Scroll to top button handler
   */
  function initScrollTop() {
      const scrollTop = document.querySelector('.scroll-top');
      if (!scrollTop) {
          console.warn('Scroll top button not found');
          return;
      }

      function toggleScrollTop() {
          scrollTop.classList.toggle('active', window.scrollY > 100);
      }

      function scrollToTop(e) {
          e.preventDefault();
          window.scrollTo({
              top: 0,
              behavior: 'smooth'
          });
      }

      scrollTop.addEventListener('click', scrollToTop);
      window.addEventListener('load', toggleScrollTop);
      document.addEventListener('scroll', toggleScrollTop);

      // Cleanup function
      return () => {
          scrollTop.removeEventListener('click', scrollToTop);
          window.removeEventListener('load', toggleScrollTop);
          document.removeEventListener('scroll', toggleScrollTop);
      };
  }

  /**
   * AOS (Animate On Scroll) initialization
   */
  function initAOS() {
      try {
          if (typeof AOS !== 'undefined') {
              AOS.init({
                  duration: 600,
                  easing: 'ease-in-out',
                  once: true,
                  mirror: false
              });
          } else {
              console.warn('AOS library not loaded');
          }
      } catch (error) {
          console.error('Error initializing AOS:', error);
      }
  }

  /**
   * Statistics counter animation
   */
  function initStatCounters() {
      const statsSection = document.querySelector('.section-stats');
      if (!statsSection) {
          console.warn('Stats section not found');
          return;
      }

      function animateCounter(element) {
          const target = parseInt(element.getAttribute('data-target'));
          if (isNaN(target)) {
              console.warn('Invalid counter target value');
              return;
          }

          const duration = 2000;
          const step = target / (duration / 16);
          let current = 0;

          const counter = setInterval(() => {
              current += step;
              if (current >= target) {
                  element.textContent = target.toLocaleString();
                  clearInterval(counter);
              } else {
                  element.textContent = Math.floor(current).toLocaleString();
              }
          }, 16);
      }

      const observer = new IntersectionObserver((entries) => {
          entries.forEach(entry => {
              if (entry.isIntersecting) {
                  const counters = entry.target.querySelectorAll('.stat-number');
                  counters.forEach(counter => animateCounter(counter));
                  observer.unobserve(entry.target);
              }
          });
      });

      observer.observe(statsSection);

      // Cleanup function
      return () => observer.disconnect();
  }

  /**
   * Navigation menu scrollspy
   */
  function initScrollSpy() {
      const navmenulinks = document.querySelectorAll('.navmenu a');
      if (!navmenulinks.length) {
          console.warn('No navigation menu links found');
          return;
      }

      function navmenuScrollspy() {
          const scrollPosition = window.scrollY + 100;
          
          navmenulinks.forEach(navmenulink => {
              if (!navmenulink.hash) return;
              
              const section = document.querySelector(navmenulink.hash);
              if (!section) return;
              
              const sectionTop = section.offsetTop - 100;
              const sectionBottom = sectionTop + section.offsetHeight;
              
              if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                  navmenulinks.forEach(link => link.classList.remove('active'));
                  navmenulink.classList.add('active');
              }
          });
      }

      window.addEventListener('load', navmenuScrollspy);
      window.addEventListener('scroll', navmenuScrollspy);

      // Cleanup function
      return () => {
          window.removeEventListener('load', navmenuScrollspy);
          window.removeEventListener('scroll', navmenuScrollspy);
      };
  }

  /**
   * Particle effect initialization
   */
  function initParticleEffect() {
      function createParticles(x, y) {
          for (let i = 0; i < 10; i++) {
              const particle = document.createElement('div');
              particle.className = 'particle';
              document.body.appendChild(particle);

              const size = Math.random() * 3 + 2;
              const destinationX = x + (Math.random() - 0.5) * 100;
              const destinationY = y + (Math.random() - 0.5) * 100;

              particle.style.width = `${size}px`;
              particle.style.height = `${size}px`;
              particle.style.left = `${x}px`;
              particle.style.top = `${y}px`;

              const animation = particle.animate([
                  {
                      transform: `translate(0, 0)`,
                      opacity: 1
                  },
                  {
                      transform: `translate(${destinationX - x}px, ${destinationY - y}px)`,
                      opacity: 0
                  }
              ], {
                  duration: Math.random() * 1000 + 500,
                  easing: 'cubic-bezier(0, .9, .57, 1)',
                  delay: Math.random() * 200
              });

              animation.onfinish = () => particle.remove();
          }
      }

      // Add click event listener for particle effect
      document.addEventListener('click', (e) => {
          createParticles(e.clientX, e.clientY);
      });
  }

  /**
   * Smooth scroll container initialization
   */
  function initSmoothScroll() {
      const scrollContainer = document.querySelector('.scroll-container');
      if (!scrollContainer) {
          console.warn('Scroll container not found');
          return;
      }

      let isDown = false;
      let startX;
      let scrollLeft;

      function handleMouseDown(e) {
          isDown = true;
          startX = e.pageX - scrollContainer.offsetLeft;
          scrollLeft = scrollContainer.scrollLeft;
          scrollContainer.style.cursor = 'grabbing';
      }

      function handleMouseLeave() {
          isDown = false;
          scrollContainer.style.cursor = 'grab';
      }

      function handleMouseUp() {
          isDown = false;
          scrollContainer.style.cursor = 'grab';
      }

      function handleMouseMove(e) {
          if (!isDown) return;
          e.preventDefault();
          const x = e.pageX - scrollContainer.offsetLeft;
          const walk = (x - startX) * 2;
          scrollContainer.scrollLeft = scrollLeft - walk;
      }

      scrollContainer.addEventListener('mousedown', handleMouseDown);
      scrollContainer.addEventListener('mouseleave', handleMouseLeave);
      scrollContainer.addEventListener('mouseup', handleMouseUp);
      scrollContainer.addEventListener('mousemove', handleMouseMove);

      // Cleanup function
      return () => {
          scrollContainer.removeEventListener('mousedown', handleMouseDown);
          scrollContainer.removeEventListener('mouseleave', handleMouseLeave);
          scrollContainer.removeEventListener('mouseup', handleMouseUp);
          scrollContainer.removeEventListener('mousemove', handleMouseMove);
      };
  }

  // Initialize everything when DOM is ready
  document.addEventListener('DOMContentLoaded', initializeWebsite);

  // Store cleanup functions for potential future use
  window._cleanup = {
      mobileNav: null,
      dropdowns: null,
      scrollTop: null,
      scrollSpy: null,
      smoothScroll: null
  };
})();


// How it Works section
document.addEventListener('DOMContentLoaded', function() {
    // Add interactive hover effects
    const stepItems = document.querySelectorAll('.step-item');
    
    stepItems.forEach(item => {
      item.addEventListener('mouseenter', function() {
        stepItems.forEach(otherItem => {
          if (otherItem !== item) {
            otherItem.style.opacity = '0.7';
            otherItem.style.transform = 'scale(0.95)';
          }
        });
      });
      
      item.addEventListener('mouseleave', function() {
        stepItems.forEach(otherItem => {
          otherItem.style.opacity = '1';
          otherItem.style.transform = 'none';
        });
      });
    });

    // Add particle effects on hover
    function createParticle(x, y) {
      const particle = document.createElement('div');
      particle.className = 'particle';
      particle.style.left = x + 'px';
      particle.style.top = y + 'px';
      particle.style.width = Math.random() * 4 + 2 + 'px';
      particle.style.height = particle.style.width;
      
      document.body.appendChild(particle);
      
      const animation = particle.animate([
        { transform: `translate(${(Math.random() - 0.5) * 100}px, ${(Math.random() - 0.5) * 100}px)`,
          opacity: 1 },
        { transform: `translate(${(Math.random() - 0.5) * 200}px, ${(Math.random() - 0.5) * 200}px)`,
          opacity: 0 }
      ], {
        duration: 1000 + Math.random() * 1000,
        easing: 'cubic-bezier(0,0,0.2,1)'
      });
      
      animation.onfinish = () => particle.remove();
    }
    
    stepItems.forEach(item => {
      item.addEventListener('mousemove', (e) => {
        const rect = item.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        if (Math.random() < 0.1) {
          createParticle(e.clientX, e.clientY);
        }
      });
    });
  });

//   article

function createFloatingElements() {
    const container = document.querySelector('.floating-elements');
    const numberOfElements = 15;
    
    for (let i = 0; i < numberOfElements; i++) {
        const element = document.createElement('div');
        element.className = 'floating-element';
        
        // Random size between 10 and 40px
        const size = Math.random() * 30 + 10;
        element.style.width = `${size}px`;
        element.style.height = `${size}px`;
        
        // Random position
        element.style.left = `${Math.random() * 100}%`;
        element.style.top = `${Math.random() * 100}%`;
        
        // Random animation
        const duration = Math.random() * 20 + 10;
        element.style.animation = `float ${duration}s infinite linear`;
        
        container.appendChild(element);
    }
}

// Animate counter numbers
function animateCounters() {
    const counters = document.querySelectorAll('.counter-number');
    counters.forEach(counter => {
        const target = parseFloat(counter.getAttribute('data-value'));
        const duration = 2000;
        const step = target / duration * 10;
        let current = 0;
        
        const updateCounter = () => {
            if (current < target) {
                current += step;
                if (current > target) current = target;
                counter.textContent = current.toFixed(current % 1 === 0 ? 0 : 1) + 
                    (counter.textContent.includes('+') ? '+' : '%');
                requestAnimationFrame(updateCounter);
            }
        };
        
        updateCounter();
    });
}


 // Accordion functionality
 document.querySelectorAll('.info-header').forEach(header => {
    header.addEventListener('click', () => {
      const content = header.parentElement.querySelector('.info-content');
      const isActive = content.classList.contains('active');
      
      // Close all content
      document.querySelectorAll('.info-content').forEach(c => {
        c.classList.remove('active');
        c.style.maxHeight = null;
      });

      // Toggle clicked content
      if (!isActive) {
        content.classList.add('active');
        content.style.maxHeight = content.scrollHeight + 'px';
      }

      // Rotate chevron
      const chevron = header.querySelector('.bi-chevron-down');
      document.querySelectorAll('.bi-chevron-down').forEach(c => {
        c.style.transform = 'rotate(0deg)';
      });
      if (!isActive) {
        chevron.style.transform = 'rotate(180deg)';
      }
    });
  });

  // Open first items by default
  document.querySelectorAll('.info-accordion').forEach(accordion => {
    const firstHeader = accordion.querySelector('.info-header');
    if (firstHeader) {
      firstHeader.click();
    }
  });
